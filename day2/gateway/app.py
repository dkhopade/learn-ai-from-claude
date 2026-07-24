"""
gateway.py — a lightweight LLM API gateway.

Sits between clients and vLLM. Responsibilities (built incrementally):
  1. Routing   — pick the right model (base vs sql-lora) per request
  2. Metrics   — emit Prometheus metrics for every request
  3. Caching   — (added next) skip the GPU on repeat prompts
  4. Rate limit— (added last) protect the GPU

This version: routing + metrics.
"""
import os
import time
import httpx
import hashlib
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ── config ────────────────────────────────────────────────────────────────
VLLM_URL = os.getenv("VLLM_URL", "http://vllm.default.svc:8000")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SQL_MODEL = os.getenv("SQL_MODEL", "sql-lora")

# ── cache ─────────────────────────────────────────────────────────────────
# Simple in-memory exact-match cache. Key = hash of (model, prompt, params).
# Only used when temperature == 0 (deterministic output), so serving a stored
# response is semantically identical to re-generating it.
CACHE: dict[str, str] = {}
CACHE_MAX_ENTRIES = 1000     # simple bound so memory can't grow unbounded

def cache_key(model: str, prompt: str, max_tokens: int) -> str:
    raw = json.dumps({"m": model, "p": prompt, "t": max_tokens}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

app = FastAPI(title="LLM Gateway")

# ── metrics ───────────────────────────────────────────────────────────────
REQUESTS = Counter(
    "gateway_requests_total", "Total gateway requests",
    ["task", "model", "outcome"],
)
LATENCY = Histogram(
    "gateway_request_duration_seconds", "Request latency through the gateway",
    ["task", "model"],
)

CACHE_EVENTS = Counter(
    "gateway_cache_events_total", "Cache hits and misses",
    ["model", "event"],          # event = hit | miss | bypass
)

# ── rate limiting ─────────────────────────────────────────────────────────
# Token bucket: allows short bursts (up to BURST) while capping sustained
# throughput at RATE requests/second. Excess requests get a fast, cheap 429
# instead of piling onto the GPU.
RATE = float(os.getenv("RATE_LIMIT_RPS", "2"))      # sustained req/sec
BURST = int(os.getenv("RATE_LIMIT_BURST", "5"))     # burst capacity

_bucket = {"tokens": float(BURST), "last": time.time()}

RATE_LIMITED = Counter(
    "gateway_rate_limited_total", "Requests rejected by the rate limiter", ["task"]
)

def allow_request() -> bool:
    now = time.time()
    elapsed = now - _bucket["last"]
    _bucket["tokens"] = min(BURST, _bucket["tokens"] + elapsed * RATE)
    _bucket["last"] = now
    if _bucket["tokens"] >= 1.0:
        _bucket["tokens"] -= 1.0
        return True
    return False

# ── request schema ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    task: str = "general"      # "general" or "sql" -> drives routing
    max_tokens: int = 256
    temperature: float = 0.0


# ── routing logic ─────────────────────────────────────────────────────────
def route_model(task: str) -> str:
    """
    Decide which model handles this request.
    Simple rule-based routing to start: 'sql' tasks -> fine-tuned adapter,
    everything else -> base model. This is where smarter routing (a
    classifier, cost-based selection) would later plug in.
    """
    if task.lower() == "sql":
        return SQL_MODEL
    return BASE_MODEL


# ── the gateway endpoint ──────────────────────────────────────────────────
@app.post("/v1/chat")
async def chat(req: ChatRequest):
    if not allow_request():
        RATE_LIMITED.labels(req.task).inc()
        raise HTTPException(status_code=429, detail="rate limit exceeded, retry later")

    model = route_model(req.task)
    start = time.time()

    # cache only makes sense for deterministic requests
    cacheable = req.temperature == 0.0
    key = cache_key(model, req.prompt, req.max_tokens) if cacheable else None

    if cacheable and key in CACHE:
        CACHE_EVENTS.labels(model, "hit").inc()
        REQUESTS.labels(req.task, model, "success").inc()
        LATENCY.labels(req.task, model).observe(time.time() - start)
        return {"model": model, "task": req.task,
                "response": CACHE[key], "cached": True}

    if cacheable:
        CACHE_EVENTS.labels(model, "miss").inc()
    else:
        CACHE_EVENTS.labels(model, "bypass").inc()

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": req.prompt}],
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                },
            )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]

        if cacheable and len(CACHE) < CACHE_MAX_ENTRIES:
            CACHE[key] = answer

        REQUESTS.labels(req.task, model, "success").inc()
        LATENCY.labels(req.task, model).observe(time.time() - start)
        return {"model": model, "task": req.task, "response": answer, "cached": False}

    except Exception as e:
        REQUESTS.labels(req.task, model, "error").inc()
        LATENCY.labels(req.task, model).observe(time.time() - start)
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

# ── Prometheus scrape endpoint ────────────────────────────────────────────
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}
