from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from tools import embed_model, qdrant, init_knowledge_base
from agent import run_agent
from llm_client import generate, LLM_MODEL

@asynccontextmanager
async def lifespan(app: FastAPI):
    count = init_knowledge_base()
    print(f"Indexed {count} chunks on startup")
    print(f"LLM backend: {os.getenv('LLM_BACKEND', 'ollama')} @ {os.getenv('LLM_BASE_URL', 'http://localhost:11434')}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend LB origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- models ---

class ChatRequest(BaseModel):
    prompt: str
    model: str = None

class SearchRequest(BaseModel):
    query: str
    limit: int = 3

class RagRequest(BaseModel):
    question: str
    model: str = None
    limit: int = 3

class AgentRequest(BaseModel):
    question: str
    session_id: str = "default"
    model: str = "llama3.1:8b"

sessions = {}

# --- helpers ---

def build_context(question: str, limit: int = 3):
    vector = embed_model.encode(question).tolist()
    results = qdrant.query_points(
        collection_name="docs",
        query=vector,
        limit=limit,
        score_threshold=0.3
    )
    chunks = results.points
    if not chunks:
        return "", []
    context = "\n\n".join([
        f"Source {i+1}: {r.payload['text']}"
        for i, r in enumerate(chunks)
    ])
    return context, chunks

def rag_prompt(context: str, question: str) -> str:
    return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so honestly.

Context:
{context}

Question: {question}

Answer:"""

# --- endpoints ---

@app.get("/")
def root():
    return {"message": "AI week 5 — deployed on OKE"}

@app.get("/health")
def health():
    count = qdrant.count(collection_name="docs")
    return {"status": "ok", "chunks_indexed": count.count}

@app.post("/search")
async def search(req: SearchRequest):
    vector = embed_model.encode(req.query).tolist()
    results = qdrant.query_points(
        collection_name="docs",
        query=vector,
        limit=req.limit,
        score_threshold=0.3
    )
    return {
        "query": req.query,
        "results": [
            {"text": r.payload["text"], "score": round(r.score, 4)}
            for r in results.points
        ]
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    answer = await generate(req.prompt, model=req.model, stream=False)
    return {"response": answer}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def gen():
        async for token in await generate(req.prompt, model=req.model, stream=True):
            yield token
    return StreamingResponse(gen(), media_type="text/plain")

@app.post("/rag")
async def rag(req: RagRequest):
    context, chunks = build_context(req.question, req.limit)
    if not context:
        return {
            "question": req.question,
            "answer": "I could not find relevant information in my knowledge base to answer that question.",
            "sources": []
        }
    answer = await generate(rag_prompt(context, req.question), model=req.model, stream=False)
    return {
        "question": req.question,
        "answer": answer,
        "sources": [r.payload["text"][:100] + "..." for r in chunks]
    }

@app.post("/rag/stream")
async def rag_stream(req: RagRequest):
    context, chunks = build_context(req.question, req.limit)
    if not context:
        async def no_context():
            yield "I could not find relevant information to answer that question."
        return StreamingResponse(no_context(), media_type="text/plain")

    import json as _json
    async def gen():
        sources = [r.payload["text"][:100] + "..." for r in chunks]
        yield _json.dumps({"sources": sources}) + "\n---\n"
        async for token in await generate(rag_prompt(context, req.question), model=req.model, stream=True):
            yield token
    return StreamingResponse(gen(), media_type="text/plain")

@app.post("/agent")
async def agent(req: AgentRequest):
    # Note: agent still uses Ollama directly via agent.py.
    # Works locally; will be ported to vLLM OpenAI tool-calling format later.
    history = sessions.get(req.session_id, [])
    result = run_agent(req.question, history=history, model=req.model)
    sessions[req.session_id] = result["messages"]
    return {"answer": result["answer"], "steps": result["steps"]}
