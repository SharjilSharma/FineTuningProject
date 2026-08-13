"""
train.py
--------
Fine-tuning script using SFTTrainer on train_bootstrap.jsonl.
"""
import os
import json
import torch
from huggingface_hub import hf_hub_download
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.finetune.config import FinetuneConfig
from src.labeling.bootstrap import SYSTEM_PROMPT

# HF dataset repo where the processed parquet is stored
_HF_DATASET_REPO = "sharjilsharma/earnings-transcripts-data"
_PARQUET_FILENAME = "phase1_it_transcripts.parquet"

def _ensure_parquet(local_path: str) -> None:
    """Download the parquet from HF Hub if it isn't already on disk."""
    if os.path.exists(local_path):
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            f"'{local_path}' not found locally and HF_TOKEN is not set. "
            "Set HF_TOKEN so the file can be downloaded from "
            f"huggingface.co/datasets/{_HF_DATASET_REPO}."
        )
    print(f"'{local_path}' not found locally — downloading from HF Hub...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=_HF_DATASET_REPO,
        filename=_PARQUET_FILENAME,
        repo_type="dataset",
        token=token,
        local_dir=os.path.dirname(local_path),
    )
    # hf_hub_download may save to a cache subdir; move to expected path if needed
    if os.path.abspath(downloaded) != os.path.abspath(local_path):
        import shutil
        shutil.move(downloaded, local_path)
    print(f"Downloaded to '{local_path}'.")

def load_dataset_for_sft(dataset_path: str):
    """
    Reads train_bootstrap.jsonl and formats it for chat/instruction tuning.
    We convert each chunk into a single conversation.
    Downloads phase1_it_transcripts.parquet from HF Hub automatically if
    it is not already present locally.
    """
    df_path = "data/processed/phase1_it_transcripts.parquet"
    _ensure_parquet(df_path)
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

    if smoke_test:
        # Smoke test: tiny randomly-initialised model on CPU — no downloads, no bitsandbytes
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
        # Production path: 4-bit QLoRA — loads model entirely on GPU via bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4 — best accuracy for LLMs
            bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16, store in 4-bit
            bnb_4bit_use_double_quant=True,     # nested quantization saves ~0.4 bits/param
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",                  # routes all layers to GPU automatically
        )
        # prepare_model_for_kbit_training is only meaningful after 4-bit loading:
        # enables gradient checkpointing and upcasts layernorm/lm_head to fp32
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
        bf16=not smoke_test,   # bf16 for compute on Colab T4/A100; False for CPU smoke test
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
