import os
import torch
from unsloth import FastLanguageModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODEL_PATH = "outputs/qwen_3b_production_best"
# You can change this to your specific repo name
HF_REPO_NAME = "khushianand01/Disposition-Qwen2.5-3B-Instruct" 

def push_model():
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_PATH,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Pushing to Hugging Face Hub: {HF_REPO_NAME}...")
    
    # Push explicit GGUF or Low-Rank adapters if needed
    # For now, pushing the merged model (or adapters depending on how it was saved)
    # If you saved just adapters, this pushes adapters. 
    # If we want to push merged:
    
    # Note: If we want to push the MERGED 16bit model (better for vLLM/Production):
    # model.push_to_hub_merged(HF_REPO_NAME, tokenizer, save_method="merged_16bit")
    
    # Pushing just the LoRA adapters (smaller, good for Unsloth inference):
    model.push_to_hub(HF_REPO_NAME, tokenizer, token=os.getenv("HF_TOKEN"))
    
    # Also valid:
    # model.push_to_hub_merged(HF_REPO_NAME, tokenizer, save_method = "merged_16bit", token = os.getenv("HF_TOKEN"))
    
    print("✅ Upload Complete!")
    print(f"View at: https://huggingface.co/{HF_REPO_NAME}")

if __name__ == "__main__":
    # Check for token
    if not os.getenv("HF_TOKEN"):
        print("⚠️  WARNING: HF_TOKEN not found in environment.")
        print("Please run: export HF_TOKEN=hf_... or add it to .env")
        # exit(1) # commenting out to let user decide
    
    try:
        push_model()
    except Exception as e:
        print(f"❌ Error: {e}")
