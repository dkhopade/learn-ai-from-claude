from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import httpx
import json

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(":memory:")

@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant.create_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    with open("docs.txt") as f:
        chunks = [c.strip() for c in f.read().split("\n\n")
                  if c.strip() and not c.startswith("#")]
    embeddings = embed_model.encode(chunks)
    points = [
        PointStruct(id=i, vector=e.tolist(),
                    payload={"text": c, "source": "docs.txt"})
        for i, (c, e) in enumerate(zip(chunks, embeddings))
    ]
    qdrant.upsert(collection_name="docs", points=points)
    print(f"Indexed {len(points)} chunks on startup")
    yield

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    prompt: str
    system: str = "You are a helpful AI assistant"
    model: str = "llama3.2:3b"

class SearchRequest(BaseModel):
    query: str
    limit: int = 3

@app.get("/")
def root():
    return {"message": "AI week 2 — embeddings + semantic search"}

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