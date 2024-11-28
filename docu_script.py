import transformers
import torch

# Model ID for Llama 3.1 8B Instruct
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load the pipeline for text generation
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},  # Ensure precision for efficiency
    device_map="auto",  # Automatically use the appropriate device (GPU or CPU)
)

# Define the chat-based messages
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Generate text
outputs = pipeline(
    messages,
    max_new_tokens=256  # Limit the length of the generated output
)

# Print the result
print(outputs[0]["generated_text"])