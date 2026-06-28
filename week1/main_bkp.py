from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: ChatRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b",
                  "prompt": req.prompt,
                  "stream": False},
            timeout=60
        )
    return {"response": resp.json()["response"]}
