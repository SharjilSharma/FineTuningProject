"""
index.py
--------
Build, persist, and query the FAISS vector index.

Responsibilities (Phase 7 — NOT implemented here)
--------------------------------------------------
- Build a flat L2 (or IVF) FAISS index from transcript embeddings.
- Persist the index to disk (.index file, gitignored).
- Provide a query function: given a new transcript embedding, return
  the k most similar historical transcripts with their stored labels.
- Index is used by retrieve_node in the LangGraph agent for grounding context.
"""
