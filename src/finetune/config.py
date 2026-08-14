"""
config.py
---------
Configuration for LoRA fine-tuning Phase 4.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class FinetuneConfig:
    # Model Choice
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # LoRA Config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training arguments
    batch_size: int = 4
    gradient_accumulation_steps: int = 2   # effective batch = batch_size × grad_accum = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    # p50=1720, p90=1839, p99=1913, max=2052 — must stay at 2048+ to avoid truncating content.
    # 2176 gives safe margin above observed max; accepts ~60s/step on T4 as a real data cost.
    max_seq_length: int = 2176
    
    # Paths
    dataset_path: str = "data/labels/train_bootstrap.jsonl"
    output_dir: str = "data/models/finetuned_adapter"
    gguf_output_dir: str = "data/models/gguf"
