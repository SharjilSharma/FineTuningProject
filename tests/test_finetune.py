import os
import pytest
from src.finetune.config import FinetuneConfig
from src.finetune.train import train

def test_finetune_config():
    cfg = FinetuneConfig()
    assert cfg.base_model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert "q_proj" in cfg.target_modules

def test_smoke_train():
    cfg = FinetuneConfig()
    # Only run the smoke test if the bootstrap dataset is actually available
    if not os.path.exists(cfg.dataset_path):
        pytest.skip("Bootstrap dataset not found. Skipping smoke test.")
        
    # Smoke test will load only 4 items and train for 2 steps
    try:
        train(smoke_test=True)
    except Exception as e:
        pytest.fail(f"Smoke test failed with error: {e}")
        
    assert os.path.exists(cfg.output_dir)
    assert os.path.exists(os.path.join(cfg.output_dir, "adapter_config.json"))
