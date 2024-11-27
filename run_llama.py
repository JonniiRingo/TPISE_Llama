import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Paths
model_path = "./consolidated.00.pth"
config_path = "./params.json"
tokenizer_path = "./tokenizer_dir"

def load_llama():
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, legacy=True)
    
    # Load the model
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config_path,
        torch_dtype=torch.float16,  # Adjust based on your hardware
    )
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

def run_inference(model, tokenizer, prompt):
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    print("Generating response...")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_llama()
    
    prompt = "Explain the theory of relativity in simple terms."
    response = run_inference(model, tokenizer, prompt)
    
    print("\nPrompt:")
    print(prompt)
    print("\nResponse:")
    print(response)