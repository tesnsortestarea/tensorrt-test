from transformers import CLIPVisionModel, CLIPProcessor
import torch
import os

# Load CLIP
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)
clip_model.eval()

output_dir = "tensorrt_quantized_models_vision"
os.makedirs(output_dir, exist_ok=True)


dummy_image = torch.zeros(1, 3, 224, 224).to(
    device)  # CLIP's default input size is 224x224
clip_onnx_path = os.path.join(output_dir, "clip_vision_encoder.onnx")

# Define dynamic axes for batch size
dynamic_axes = {
    'pixel_values': {0: 'batch_size'},
    # CLIP's output is usually last_hidden_state or pooler_output
    'last_hidden_state': {0: 'batch_size'}
}

print(f"Exporting CLIP to ONNX at {clip_onnx_path}...")
torch.onnx.export(
    clip_model,
    dummy_image,
    clip_onnx_path,
    input_names=['pixel_values'],
    output_names=['last_hidden_state'],  # Adjust if you need pooler_output
    dynamic_axes=dynamic_axes,
    opset_version=17,  # Use a recent opset version
    do_constant_folding=True,
    verbose=False
)
print("CLIP ONNX export complete.")
