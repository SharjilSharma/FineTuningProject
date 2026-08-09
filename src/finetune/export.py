"""
export.py
---------
Merge LoRA adapter weights into the base model and export to GGUF.

Responsibilities (Phase 4 — NOT implemented here)
--------------------------------------------------
- Load base model + trained LoRA adapter.
- Merge adapter into base weights (merge_and_unload).
- Save merged model in HF format.
- Convert to GGUF 4-bit using llama.cpp's convert script for local CPU
  inference via Ollama.
- Optionally push adapter-only weights back to HF Hub (private repo).
"""
