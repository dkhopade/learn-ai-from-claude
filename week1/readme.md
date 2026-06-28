# Week 1 — FastAPI + Ollama Streaming API

## Running locally
```bash
source venv/bin/activate
uvicorn main:app --reload
```

## Endpoints
GET  /            - hello world
POST /chat        - direct LLM call
POST /chat/stream - streaming LLM call

## Note on containers
Docker is blocked by corporate policy on this machine.
Will revisit containerization in Week 5 when deploying to OCI/OKE,
where containers can be built in CI/CD pipeline instead.