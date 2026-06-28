# Week 2 — Embeddings + Vector Database + Semantic Search

## Architecture
docs.txt -> chunker -> embeddings (all-MiniLM-L6-v2) -> Qdrant -> /search endpoint

## Running locally
```bash
source venv/bin/activate
uvicorn main:app --reload
```

## Standalone scripts
- `embed_test.py`   - generate embeddings + cosine similarity demo
- `vectordb_test.py` - Qdrant collection + semantic search demo
- `indexer.py`       - load docs.txt, chunk, index, and search

## Endpoints
GET  /            - hello world
POST /search      - semantic search over docs.txt
POST /chat        - direct LLM call
POST /chat/stream - streaming LLM call

## Key concepts
- Embedding: text converted to a 384-dimension vector of meaning
- Cosine similarity: measures how close two vectors are (1.0 = identical, 0 = unrelated)
- Score threshold (0.3): filters out irrelevant results