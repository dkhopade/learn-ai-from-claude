# Week 3 — Full RAG Application

## Architecture
docs.txt -> chunker -> embeddings -> Qdrant -> /rag endpoint -> Ollama LLM -> React UI

## Running locally
Terminal 1: ollama serve
Terminal 2: source venv/bin/activate && uvicorn main:app --reload
Terminal 3: cd chat-ui && npm start

## Endpoints
GET  /health      - health check + chunk count
POST /search      - semantic search only
POST /rag         - full RAG (retrieve + generate)
POST /rag/stream  - streaming RAG
POST /chat        - direct LLM (no RAG)
POST /chat/stream - streaming direct LLM

## Frontend
React chat UI in `chat-ui/` — streams answers token by token with source citations.