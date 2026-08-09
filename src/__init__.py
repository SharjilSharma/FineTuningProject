"""
Earnings Call Signal Extraction Agent — source root.

Sub-packages
------------
data        : dataset loading, price fetching, preprocessing
baselines   : lexicon and base-model baselines
labeling    : schema definition and annotation tooling
finetune    : LoRA/QLoRA training pipeline
eval        : extraction-quality and price-correlation metrics
agent       : LangGraph multi-step orchestration agent
faiss_store : FAISS vector index build and query
api         : FastAPI serving layer
"""
