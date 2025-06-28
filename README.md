# Preparing TensorRT-Triton Server for Language-Based Inference

This guide provides a comprehensive overview of setting up a TensorRT-Triton server for language-based inference, specifically leveraging the latest GPT models from TensorRT-LLM. We'll cover the essential steps to build model engines and configure the server to handle inference requests, including the quantization pipeline. This guide focuses on practical steps rather than delving deep into the theoretical details of each component.

---

## 1. Download GPT2 Model from Hugging Face

We'll start by downloading a pre-trained GPT2 model from Hugging Face. Pre-trained models like these are invaluable as they have learned intricate patterns from vast datasets, enabling developers to utilize advanced AI capabilities without the extensive time and resources required for training complex models from scratch.

**Model Link:** [https://huggingface.co/openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium)

---

## 2. Download and Build TensorRT-LLM

TensorRT-LLM is built upon NVIDIA's TensorRT deep learning optimizer, specifically designed for Large Language Models (LLMs). It offers a Python API to define LLMs and compile them into highly optimized "TensorRT engines." These engines incorporate cutting-edge optimizations such as:

* Custom CUDA Kernels
* Quantization
* Layer Fusion
* In-Flight Batching
* Paged Attention
* Speculative Decoding
* Multi-GPU and Multi-Node Support

**Note:** You can skip building the TensorRT-LLM container in this step, as it will be included in **Step 3** when building the `tensorrtllm-backend` container.

**Source Code and Guide:** [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

**Building Docker:** [https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html)

---

## 3. Download and Build TensorRT-LLM Backend Container

The `tensorrtllm_backend` is a specialized plugin for NVIDIA Triton Inference Server, enabling it to efficiently serve LLMs optimized with NVIDIA TensorRT-LLM. It's a crucial component for deploying high-performance, scalable LLM inference solutions on NVIDIA GPUs.

This `tensorrtllm-backend` build process includes TensorRT-LLM as a dependency, so you can skip **Step 2** and simply select the correct tag and branch for TensorRT-LLM within the `tensorrtllm-backend` repository.

**Caution:** It is critical that the **version of Triton Inference Server matches the `tensorrtllm-backend` version.**

**Source and Container Build Guide:** [https://github.com/triton-inference-server/tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend)

---

## 4. Download and Build Triton Inference Server Container

NVIDIA Triton Inference Server is an open-source serving software developed by NVIDIA, designed to streamline the deployment and execution of AI models in production environments. It simplifies serving machine learning and deep learning models at scale.

**Source and Container Build Guide:** [https://github.com/triton-inference-server/server](https://github.com/triton-inference-server/server)

---

## 5. Convert Weights from Hugging Face Transformers to TensorRT-LLM Format

To convert the Hugging Face Transformers GPT model to TensorRT-LLM format, use the `convert_checkpoint.py` script located in the TensorRT-LLM examples. For the `model_dir` value, use the path `/prepare-triton-server/input-engine`.

**Script Location:** [https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/gpt/README.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/gpt/README.md)

---

## 6. Build TensorRT Engines

Using the output from **Step 5**, generate the GPT engine via the `trtllm-build` command from the TensorRT-LLM repository. This command allows for various configurations, such as `dtype` for data type and `tp` for tensor parallelism, accommodating different hardware architectures. The output folder, for example, `/prepare-triton-server/output-engine`, will be used later within the Docker container to prepare the model for inference.

**Command Description for GPT Model:** [https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/gpt/README.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/gpt/README.md)

---

## 7. Prepare Model for Inference

Copy the output from **Step 6** (the `/prepare-triton-server/output-engine` folder) to the `/all_models/inflight_batcher_llm/tensorrt_llm/1/` directory within the container so that Triton Server can locate it.

---

## 8. Prepare Configuration Files for Inference Steps

This step involves preparing the `.pbtxt` configuration files for various stages of inference, including preprocessing, ensemble, `tensorrt_llm`, and post-processing. Each LLM model requires its own configuration files, tailored to the number of inputs and outputs at each stage, with correct data types.

While most models in the TensorRT-LLM repository don't have default configuration files, the GPT model's configuration file is available at: [https://github.com/NVIDIA/TensorRT-LLM/tree/main/triton_backend/all_models/gpt](https://github.com/NVIDIA/TensorRT-LLM/tree/main/triton_backend/all_models/gpt).

It's necessary to modify certain parameters within the `.pbtxt` files for each stage. You can either use the `fill_template.py` script or manually update the values. Parameters to adapt are typically in the `${}` format and should be replaced with valid values. The location of these configuration files within the container is `/all_models/inflight_batcher_llm/tensorrt_llm`.

For more information on Triton Inference Server, refer to:

* [https://github.com/triton-inference-server/tutorials](https://github.com/triton-inference-server/tutorials)
* [https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html#launch-triton-tensorrt-llm-container](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html#launch-triton-tensorrt-llm-container)

---

## 9. Run the Triton Server

After preparing the model and configuration files, you can now run the Triton Server to accept input and provide output.

```bash
python3 /opt/scripts/launch_triton_server.py --model_repo <path/to/model>
Expected Directory Structure
After generating the engines and configuring the files, your directory structure should resemble the following:

/all_models/inflight_batcher_llm/tensorrt_llm/
├── ensemble/
│   └── 1/
│       └── config.pbtxt
├── preprocessor/
│   └── 1/
│       ├── config.pbtxt
│       ├── model.py
│       └── requirements.txt
├── tensorrt_llm/
│   └── 1/
│       ├── config.pbtxt
│       ├── rank0.engine  # Path to engine from Step 6
│       └── config.json
└── postprocessor/
    └── 1/
        ├── config.pbtxt
        ├── model.py
        └── requirements.txt
Preparing Triton Server for Visual Language Model (VLM) Inference
Building upon our knowledge of handling language-based input with Triton Server, we will now extend this principle to manage both visual and textual input for Visual Language Models (VLMs). Our goal is to leverage a Large Multimodal Model (LMM) capable of understanding and generating content based on both images and text. We will utilize the CLIP model for the vision component and MiniLM-Tex for the text component. We'll need to generate TensorRT engines for both of these models.

1. Prepare CLIP Vision and Text Engines
After downloading both models, you can generate ONNX files to standardize the neural network representation. Use the provided scripts: prepare-triton-server/engines/vision_encoder/vision_encoder.py and prepare-triton-server/engines/text_encoder/text_encoder.py to generate these ONNX files. Subsequently, trtexec can be used to generate TensorRT engines, with various parameters available for quantization.

You'll also need to create two config.pbtxt files for both the text and vision engines, which will be used later by the TensorRT-Triton Server.

/prepare-triton-server/engines/
├── vision_encoder/tensorrt_quantized_models/engines/
│   ├── config.pbtxt
│   └── 1/
│       └── clip_vision_encoder_quantized_int8.engine
├── text_encoder/tensorrt_quantized_models/engines/
│   ├── config.pbtxt
│   └── 1/
│       └── llava_llm_quantized_engine/
└── llava_ensemble/
    └── config.pbtxt
2. Ensemble Vision and Text Engines into a VLM Model
With both vision and text engines, and their respective .pbtxt configuration files in place, we can construct an ensemble to handle both text and vision as input, ultimately providing text output to be forwarded to a database for finding the best match.

/prepare-triton-server/engines/
└── llava_ensemble/
    └── config.pbtxt
3. Mocking Mongo-DB and Vector-DB
After the VLM inference, we will query a database to find the best match to send an answer to the client.

/prepare-triton-server/database/
├── mock-mongo.py
└── mock-vector.py
