import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Use MPS if available
device = torch.device("mps" if torch.has_mps else "cpu")
print(f"Using device: {device}")

# Define the model and tokenizer paths
model_weights_path = "/Users/jonathansamuels/LLM's/transformers/Llama/Meta-Llama-3.1-8B-Instruct/original/consolidated.00.pth"
tokenizer_path = "/Users/jonathansamuels/LLM's/transformers/Llama/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model"
config_path = "/Users/jonathansamuels/LLM's/transformers/Llama/Meta-Llama-3.1-8B-Instruct/original/params.json"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

# Load model weights
print("Loading model weights...")
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=None,
    state_dict=torch.load(model_weights_path, map_location=device),
    config=config_path
).to(device)

# Example text generation
input_text = "Hello, I am a large language model."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate a response
print("Generating text...")
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=100)

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the result
print(f"Generated Text: {generated_text}")