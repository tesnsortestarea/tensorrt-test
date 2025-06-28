import tritonclient.grpc as grpcclient
import numpy as np
from database.mock-mongo import *
from database.mock-vector import *


TRITON_SERVER_URL = "localhost:8001"
MODEL_NAME = "your_tensorrt_model_name"
INPUT_NAME = "input_tensor_name"  # As defined in model's config.pbtxt
OUTPUT_NAME = "output_tensor_name"  # As defined in model's config.pbtxt

try:
    # Create Triton client
    client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)

    input_data = ["reseived from client"]

    # Create inference input object
    infer_input = grpcclient.InferInput(INPUT_NAME, input_data.shape, "FP32")
    infer_input.set_data_from_numpy(input_data)

    # Create inference output object
    infer_output = grpcclient.InferRequestedOutput(OUTPUT_NAME)
    # Send inference request
    response = client.infer(
        model_name=MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output]
    )

    # Get the output as a NumPy array
    output_data = response.as_numpy(OUTPUT_NAME)

    print(f"Triton Inference Output Shape: {output_data.shape}")
    print(f"Triton Inference Output Data: {output_data}")

    # Now, process output_data and bind to database

except Exception as e:
    print(f"Error communicating with Triton: {e}")
