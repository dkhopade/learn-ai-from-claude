from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import httpx
import json

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    system: str = "You are a helpful AI assistant"
    model: str = "llama3.2:3b"

@app.get("/")
def root():
    return {"message": "AI week 1 — hello world"}

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
