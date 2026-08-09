"""
embeddings.py
-------------
Generate embeddings for transcript chunks using sentence-transformers.

Responsibilities (Phase 7 — NOT implemented here)
--------------------------------------------------
- Load a sentence-transformer model (e.g. BAAI/bge-small-en-v1.5).
- Encode transcript chunks in batches.
- Cache embeddings in Redis (keyed by transcript_id + chunk_hash) to
  avoid redundant computation.
- Return numpy arrays ready for FAISS indexing.
"""
