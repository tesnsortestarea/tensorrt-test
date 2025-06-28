from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import os

from numpy.core.multiarray import *

text_encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
st_model = SentenceTransformer(text_encoder_name)

# Access the underlying Hugging Face model and tokenizer
text_model = AutoModel.from_pretrained(text_encoder_name)
text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)

text_model.to("cuda")
text_model.eval()

output_dir = "tensorrt_quantized_models_text"
os.makedirs(output_dir, exist_ok=True)


max_seq_length = 128
dummy_text_input = text_tokenizer("This is a test sentence for ONNX export.",
                                  return_tensors="pt", max_length=max_seq_length, padding="max_length", truncation=True)
dummy_text_input = {k: v.to("cuda") for k, v in dummy_text_input.items()}

text_onnx_path = os.path.join(output_dir, "minilm_text_encoder.onnx")

dynamic_axes_text = {
    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
    'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
}

print(f"Exporting MiniLM to ONNX at {text_onnx_path}...")
torch.onnx.export(
    text_model,
    tuple(dummy_text_input.values()),
    text_onnx_path,
    input_names=list(dummy_text_input.keys()),
    # The pooled output might be needed for sentence embeddings
    output_names=['last_hidden_state'],
    dynamic_axes=dynamic_axes_text,
    opset_version=17,
    do_constant_folding=True,
    verbose=False
)
print("MiniLM ONNX export complete.")
