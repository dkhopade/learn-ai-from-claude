import os
import json
import httpx

# Backend selection via env var.
# Local dev: LLM_BACKEND=ollama, LLM_BASE_URL=http://localhost:11434
# In cluster: LLM_BACKEND=vllm, LLM_BASE_URL=http://vllm:8000
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")


async def generate(prompt: str, model: str = None, stream: bool = False):
    """Non-streaming or streaming text generation, backend-agnostic.
    Returns a string (non-stream) or async-yields token strings (stream)."""
    model = model or LLM_MODEL

    if LLM_BACKEND == "vllm":
        url = f"{LLM_BASE_URL}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
    else:  # ollama
        url = f"{LLM_BASE_URL}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": stream}

    if not stream:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=120)
        data = resp.json()
        if LLM_BACKEND == "vllm":
            return data["choices"][0]["message"]["content"]
        return data["response"]
    else:
        return _stream_tokens(url, payload)


async def _stream_tokens(url: str, payload: dict):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload, timeout=120) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                if LLM_BACKEND == "vllm":
                    # vLLM streams SSE lines like: "data: {json}"
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            break
                        try:
                            delta = json.loads(chunk)["choices"][0]["delta"]
                            yield delta.get("content", "")
                        except (json.JSONDecodeError, KeyError):
                            continue
                else:  # ollama
                    try:
                        yield json.loads(line).get("response", "")
                    except json.JSONDecodeError:
                        continue
