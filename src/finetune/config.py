"""
config.py
---------
Training hyperparameters and LoRA configuration.

Responsibilities (Phase 4 — NOT implemented here)
--------------------------------------------------
- Dataclass / Pydantic model holding:
    base_model_id   : HF model repo string (e.g. "meta-llama/Llama-3.2-3B")
    lora_r          : LoRA rank
    lora_alpha      : LoRA alpha scaling
    lora_target_modules : list of modules to apply LoRA to
    load_in_4bit    : bool — QLoRA flag
    max_seq_length  : int
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate   : float
    output_dir      : path to save adapter weights
- Sensible defaults tuned for a free T4 GPU (Colab / Kaggle).
"""
