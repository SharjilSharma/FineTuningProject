"""
nodes.py
--------
Individual node functions for the LangGraph agent graph.

Responsibilities (Phase 7 — NOT implemented here)
--------------------------------------------------
Each function corresponds to one node in graph.py:
    trigger_node    : poll API Ninjas or accept a manual transcript input
    retrieve_node   : query FAISS store for k nearest historical embeddings
    extract_node    : call fine-tuned model (via Ollama) and parse JSON output
    validate_node   : validate output against Pydantic schema + confidence threshold
    store_node      : write to PostgreSQL; update running backtest metrics
"""
