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
    batch_size: int = 2
    gradient_accumulation_steps: int = 4   # effective batch = batch_size × grad_accum = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    # 2048 accommodates full earnings chunks but is slow on T4 (~60s/step).
    # Lower to 1024 to cut step time ~4x; raise back to 2048 for a final production run.
    max_seq_length: int = 1024
    
    # Paths
    dataset_path: str = "data/labels/train_bootstrap.jsonl"
    output_dir: str = "data/models/finetuned_adapter"
    gguf_output_dir: str = "data/models/gguf"
