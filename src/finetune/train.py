"""
train.py
--------
Fine-tuning script using SFTTrainer on train_bootstrap.jsonl.
"""
import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.finetune.config import FinetuneConfig
from src.labeling.bootstrap import SYSTEM_PROMPT

def load_dataset_for_sft(dataset_path: str):
    """
    Reads train_bootstrap.jsonl and formats it for chat/instruction tuning.
    We convert each chunk into a single conversation.
    """
    df_path = "data/processed/phase1_it_transcripts.parquet"
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"{df_path} missing. Need this for the chunk_text.")
    df = pd.read_parquet(df_path)
    text_map = dict(zip(df['chunk_id'], df['chunk_text']))
    
    data = []
    with open(dataset_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            cid = row.get("chunk_id")
            chunk_text = text_map.get(cid)
            if not chunk_text:
                continue
                
            prompt = f"Analyze this excerpt and return ONLY the JSON signal:\n\n--- EXCERPT ---\n{chunk_text[:6000]}\n--- END ---"
            
            label = {
                "guidance_direction": row.get("guidance_direction"),
                "tone": row.get("tone"),
                "hedging_score": row.get("hedging_score"),
                "key_flags": row.get("key_flags", [])
            }
            output_json = json.dumps(label)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output_json}
            ]
            data.append({"messages": messages})
            
    return Dataset.from_list(data)

def train(smoke_test=False):
    cfg = FinetuneConfig()
    
    print(f"Loading dataset from {cfg.dataset_path}")
    dataset = load_dataset_for_sft(cfg.dataset_path)
    
    if smoke_test:
        print("Smoke test enabled: restricting dataset size.")
        dataset = dataset.select(range(min(4, len(dataset))))
        
    print(f"Loading tokenizer for {cfg.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading base model {cfg.base_model_name}")
    
    # In smoke test (local), we don't use 4bit/bf16 to avoid bitsandbytes windows issues
    dtype = torch.float32 if smoke_test else torch.bfloat16
    
    if smoke_test:
        from transformers import AutoConfig
        print(f"Smoke test: creating tiny random version of {cfg.base_model_name}")
        config = AutoConfig.from_pretrained(cfg.base_model_name)
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        max_length=cfg.max_seq_length,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=10 if not smoke_test else 1,
        max_steps=2 if smoke_test else -1,
        num_train_epochs=cfg.num_train_epochs if not smoke_test else 1,
        save_strategy="epoch",
        fp16=False,
        bf16=not smoke_test,
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    
    if not smoke_test:
        trainer.model.print_trainable_parameters()
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final adapter to {cfg.output_dir}")
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Done!")

if __name__ == "__main__":
    train()
