"""
export.py
---------
Merges the LoRA adapter into the base model and saves it.
(GGUF conversion is usually done via llama.cpp python scripts).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.finetune.config import FinetuneConfig
import os

def merge_and_export():
    cfg = FinetuneConfig()
    
    repo_id = "sharjilsharma/earnings-signal-lora-adapter"
        
    print(f"Loading base model {cfg.base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    
    print(f"Loading adapter from Hub ({repo_id})...")
    model = PeftModel.from_pretrained(
        base_model, 
        repo_id,
        token=os.environ.get("HF_TOKEN")
    )
    
    print("Merging adapter...")
    model = model.merge_and_unload()
    
    merged_dir = cfg.output_dir + "_merged"
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        token=os.environ.get("HF_TOKEN")
    )
    tokenizer.save_pretrained(merged_dir)
    print("Merge complete! To create a GGUF file for local CPU inference, run the standard llama.cpp convert script on this directory.")

if __name__ == "__main__":
    merge_and_export()
