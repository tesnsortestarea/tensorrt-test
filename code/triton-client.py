# triton_client_app_llm.py

from database.mock-mongo import *
from database.mock-vector import *

import tritonclient.grpc as grpcclient
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Union, Optional
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import datetime  # Import for MongoDB serialization

# --- Configuration ---
TRITON_SERVER_URL = "localhost:8001"
# IMPORTANT: Match your deployed LLM embedding model
LLM_EMBEDDING_MODEL_NAME = "your_llm_embedding_model_name"
LLM_INPUT_IDS_NAME = "input_ids"
LLM_ATTENTION_MASK_NAME = "attention_mask"
LLM_OUTPUT_EMBEDDING_NAME = "embedding_output"

EMBEDDING_DIM = 768
FAISS_DB_PATH = "faiss_db/my_product_embeddings"
MOCK_MONGODB_PATH = "mock_db"

# Multi-threading configuration
NUM_INFERENCE_THREADS = 4  # Threads to send requests and handle callbacks
# Threads to process results (FAISS search, DB query)
NUM_POST_PROCESSING_THREADS = 4
MAX_REQUEST_QUEUE_SIZE = 100  # Max pending requests to Triton

# --- Global Instances ---
vector_db: Optional[FAISSVectorDB] = None
mongo_db: Optional[MockMongoDB] = None
tokenizer: Optional[AutoTokenizer] = None

# Thread-safe queue for Triton responses
triton_response_queue = queue.Queue()

# Thread-safe queue for client responses to be delivered back
# Where processed results are placed for the client to poll
final_client_response_queue = queue.Queue()

# Unique ID generator for requests
request_id_counter = 0
request_id_lock = threading.Lock()


def get_next_request_id():
    global request_id_counter
    with request_id_lock:
        req_id = request_id_counter
        request_id_counter += 1
        return str(req_id)  # Return as string for consistency


def load_faiss_db():
    global vector_db
    if vector_db is None:
        print(f"Loading FAISS DB from {FAISS_DB_PATH}...")
        try:
            vector_db = FAISSVectorDB.load(FAISS_DB_PATH)
            print(
                f"FAISS DB loaded successfully with {vector_db.next_id} products.")
        except FileNotFoundError:
            print(
                f"Error: FAISS DB files not found at {FAISS_DB_PATH}. Please run precompute_and_populate.py first.")
            exit(1)
        except Exception as e:
            print(f"Error loading FAISS DB: {e}")
            exit(1)


def load_mongo_db():
    global mongo_db
    if mongo_db is None:
        print(f"Initializing Mock MongoDB at {MOCK_MONGODB_PATH}...")
        try:
            mongo_db = MockMongoDB(db_path=MOCK_MONGODB_PATH)
            print("Mock MongoDB initialized.")
        except Exception as e:
            print(f"Error initializing Mock MongoDB: {e}")
            exit(1)


def load_tokenizer():
    global tokenizer
    if tokenizer is None:

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/MiniLM")
        except Exception as e:
            print(
                f"Error loading tokenizer: {e}. Ensure 'transformers' is installed and path is correct.")
            exit(1)


# Preprocessing for LLM
def preprocess_text_for_llm(text: str, max_length: int = 128) -> Dict[str, np.ndarray]:
    """Tokenizes text for a TensorRT-LLM model."""
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded.")

    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    for key in inputs:
        inputs[key] = inputs[key].astype(np.int32)
    return {
        LLM_INPUT_IDS_NAME: inputs["input_ids"],
        LLM_ATTENTION_MASK_NAME: inputs["attention_mask"],
    }


# Triton Client Callback
def callback(result, error, request_context):
    """
    Callback function executed by Triton client when an async inference completes.
    It puts the result into a queue for post-processing.
    """
    if error:
        print(
            f"Inference error for request {request_context.get('request_id')}: {error}")
        triton_response_queue.put(
            {"error": str(error), "request_id": request_context.get('request_id')})
    else:
        triton_response_queue.put(
            {"result": result, "request_id": request_context.get('request_id')})


# Inference Thread Function
def inference_worker(triton_client: grpcclient.InferenceServerClient, request_queue: queue.Queue):
    """
    Worker function for sending inference requests to Triton.
    """
    while True:
        request_data = request_queue.get()
        if request_data is None:  # Sentinel value to stop thread
            break

        query_text = request_data["query_text"]
        # The original request ID from the client
        client_request_id = request_data["client_request_id"]

        try:
            # Getting input for llm model
            preprocessed_inputs = preprocess_text_for_llm(query_text)

            triton_inputs = []
            for input_name, input_data in preprocessed_inputs.items():
                if input_data is not None:
                    triton_inputs.append(grpcclient.InferInput(
                        input_name, input_data.shape, "INT32"))
                    triton_inputs[-1].set_data_from_numpy(input_data)

            infer_output = grpcclient.InferRequestedOutput(
                LLM_OUTPUT_EMBEDDING_NAME)

            # Send asynchronous inference request
            triton_client.async_infer(
                model_name=LLM_EMBEDDING_MODEL_NAME,
                inputs=triton_inputs,
                callback=callback,
                # Put interence resutl into Queue
                outputs=[infer_output],
                request_id=client_request_id,  # Pass the client's request_id to the callback context
                client_timeout=120.0
            )

        except Exception as e:
            print(
                f"Error sending request for client_request_id {client_request_id} to Triton: {e}")
            # Put an error response directly into the response queue if send fails
            triton_response_queue.put(
                {"error": f"Failed to send request: {e}", "request_id": client_request_id})
        finally:
            request_queue.task_done()


# Post-Processing Thread Function
def post_processing_worker(k: int = 5):
    """
    Worker function to process Triton responses, search FAISS, and query MongoDB.
    """
    while True:
        response_item = triton_response_queue.get()
        if response_item is None:  # Sentinel value to stop thread
            break

        request_id = response_item.get("request_id")
        final_result = {"request_id": request_id}

        if "error" in response_item:
            final_result["status"] = "error"
            final_result["message"] = response_item["error"]
            print(
                f"Processing error for request {request_id}: {response_item['error']}")
        else:
            try:
                # Extract embedding from Triton response
                triton_result = response_item["result"]
                embedding = triton_result.as_numpy(LLM_OUTPUT_EMBEDDING_NAME)
                if embedding is None:
                    raise ValueError(
                        f"Triton returned None for output '{LLM_OUTPUT_EMBEDDING_NAME}'")

                # Ensure embedding is 1D and normalized
                if embedding.shape[0] != 1 or embedding.shape[-1] != EMBEDDING_DIM:
                    raise ValueError(
                        f"Unexpected embedding shape: {embedding.shape}")
                normalized_embedding = embedding[0] / \
                    np.linalg.norm(embedding[0])

                # Search FAISS
                faiss_results = vector_db.search(normalized_embedding, k=k)

                # Retrieve full product details from MongoDB
                detailed_results = []
                for product_id, similarity_distance in faiss_results:
                    mongo_product_details = mongo_db.find_product_by_id(
                        product_id)
                    if mongo_product_details:
                        # Clean up binary data for printing if it exists
                        if "image" in mongo_product_details and isinstance(mongo_product_details["image"], bytes):
                            mongo_product_details["image"] = "bytes_data_omitted_for_print"
                        detailed_results.append({
                            "product_id": product_id,
                            "similarity_distance": float(similarity_distance),
                            "details": mongo_product_details
                        })
                    else:
                        print(
                            f"Warning: Product ID {product_id} not found in Mock MongoDB for request {request_id}.")

                final_result["status"] = "success"
                final_result["results"] = detailed_results

            except Exception as e:
                final_result["status"] = "error"
                final_result["message"] = f"Post-processing failed: {e}"
                print(f"Post-processing error for request {request_id}: {e}")

        final_client_response_queue.put(final_result)
        triton_response_queue.task_done()


# Main Application Logic
def run_client_application():
    """
    Sets up the multi-threaded client, simulates incoming requests,
    and processes results.
    """
    load_tokenizer()
    load_faiss_db()
    load_mongo_db()

    # Create Triton client (async_req=True is essential for callbacks)
    triton_client = grpcclient.InferenceServerClient(
        url=TRITON_SERVER_URL,
        verbose=False,  # Set to True for more verbose logging from tritonclient
        async_req=True
    )

    # Queue for requests to be sent to Triton
    inference_request_queue = queue.Queue(maxsize=MAX_REQUEST_QUEUE_SIZE)

    # Start inference worker threads
    inference_workers = []
    for _ in range(NUM_INFERENCE_THREADS):
        t = threading.Thread(target=inference_worker, args=(
            triton_client, inference_request_queue))
        t.daemon = True  # Allows main program to exit even if threads are running
        t.start()
        inference_workers.append(t)

    # Start post-processing worker threads
    post_processing_workers = []
    for _ in range(NUM_POST_PROCESSING_THREADS):
        t = threading.Thread(target=post_processing_worker,
                             args=(5,))  # k=5 for search
        t.daemon = True
        t.start()
        post_processing_workers.append(t)

    print(
        f"\nClient application running with {NUM_INFERENCE_THREADS} inference threads and {NUM_POST_PROCESSING_THREADS} post-processing threads.")
    print("Simulating client requests...")


if __name__ == "__main__":

    os.makedirs("faiss_db", exist_ok=True)
    os.makedirs("mock_db", exist_ok=True)

    print("Running precompute_and_populate.py to set up databases...")
    import subprocess
    try:
        subprocess.run(["python", "precompute_and_populate.py"], check=True)
        print("Databases populated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error populating databases: {e}")
        exit(1)
    except FileNotFoundError:
        print("Error: precompute_and_populate.py not found. Make sure it's in the same directory.")
        exit(1)

    run_client_application()
