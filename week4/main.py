from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json

from tools import embed_model, qdrant, init_knowledge_base
from agent import run_agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    count = init_knowledge_base()
    print(f"Indexed {count} chunks on startup")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- models ---

class ChatRequest(BaseModel):
    prompt: str
    system: str = "You are a helpful AI assistant"
    model: str = "llama3.2:3b"

class SearchRequest(BaseModel):
    query: str
    limit: int = 3

class RagRequest(BaseModel):
    question: str
    model: str = "llama3.2:3b"
    limit: int = 3

class AgentRequest(BaseModel):
    question: str
    session_id: str = "default"
    model: str = "llama3.1:8b"

# --- session store for agent memory ---

sessions = {}

# --- helpers ---

def build_context(question: str, limit: int = 3) -> tuple:
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

# --- endpoints ---

@app.get("/")
def root():
    return {"message": "AI week 4 — agents and tool calling"}

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
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": req.model,
                  "prompt": req.prompt,
                  "stream": False},
            timeout=60
        )
    return {"response": resp.json()["response"]}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/generate",
                json={"model": req.model,
                      "prompt": req.prompt,
                      "stream": True},
                timeout=60
            ) as r:
                async for line in r.aiter_lines():
                    if line:
                        data = json.loads(line)
                        yield data.get("response", "")
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/rag")
async def rag(req: RagRequest):
    context, chunks = build_context(req.question, req.limit)

    if not context:
        return {
            "question": req.question,
            "answer": "I could not find relevant information in my knowledge base to answer that question.",
            "sources": []
        }

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so honestly.

Context:
{context}

Question: {req.question}

Answer:"""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": req.model,
                  "prompt": prompt,
                  "stream": False},
            timeout=120
        )

    return {
        "question": req.question,
        "answer": resp.json()["response"],
        "sources": [r.payload["text"][:100] + "..." for r in chunks]
    }

@app.post("/rag/stream")
async def rag_stream(req: RagRequest):
    context, chunks = build_context(req.question, req.limit)

    if not context:
        async def no_context():
            yield "I could not find relevant information to answer that question."
        return StreamingResponse(no_context(), media_type="text/plain")

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so honestly.

Context:
{context}

Question: {req.question}

Answer:"""

    async def generate():
        sources = [r.payload["text"][:100] + "..." for r in chunks]
        yield json.dumps({"sources": sources}) + "\n---\n"
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/generate",
                json={"model": req.model,
                      "prompt": prompt,
                      "stream": True},
                timeout=120
            ) as r:
                async for line in r.aiter_lines():
                    if line:
                        data = json.loads(line)
                        yield data.get("response", "")

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/agent")
async def agent(req: AgentRequest):
    history = sessions.get(req.session_id, [])
    result = run_agent(req.question, history=history, model=req.model)
    sessions[req.session_id] = result["messages"]
    return {
        "answer": result["answer"],
        "steps": result["steps"]
    }
