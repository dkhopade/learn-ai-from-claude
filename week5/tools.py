import httpx
import os
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# qdrant = QdrantClient(":memory:")

QDRANT_URL = os.getenv("QDRANT_URL", "")

if QDRANT_URL:
    # cluster: connect to the real Qdrant service
    qdrant = QdrantClient(url=QDRANT_URL)
else:
    # local dev: in-memory, no external Qdrant needed
    qdrant = QdrantClient(":memory:")

def init_knowledge_base(max_retries=10, retry_delay=6):
    # wait for Qdrant to be reachable before proceeding
    for attempt in range(1, max_retries + 1):
        try:
            # a lightweight call to check Qdrant is up
            qdrant.get_collections()
            break
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Qdrant not reachable after {max_retries} attempts: {e}"
                )
            print(f"Qdrant not ready (attempt {attempt}/{max_retries}), "
                  f"retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    qdrant.recreate_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    with open("docs.txt") as f:
        chunks = [c.strip() for c in f.read().split("\n\n")
                  if c.strip() and not c.startswith("#")]
    embeddings = embed_model.encode(chunks)
    points = [
        PointStruct(id=i, vector=e.tolist(), payload={"text": c})
        for i, (c, e) in enumerate(zip(chunks, embeddings))
    ]
    qdrant.upsert(collection_name="docs", points=points)
    return len(points)

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    allowed = "0123456789+-*/(). "
    if all(c in allowed for c in expression):
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"
    return "Invalid expression"

def get_weather(latitude: float, longitude: float) -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": latitude, "longitude": longitude,
              "current": "temperature_2m,wind_speed_10m"}
    r = httpx.get(url, params=params, timeout=10)
    data = r.json()["current"]
    return f"Temperature: {data['temperature_2m']}°C, Wind: {data['wind_speed_10m']} km/h"

def search_knowledge_base(query: str) -> str:
    vector = embed_model.encode(query).tolist()
    results = qdrant.query_points(
        collection_name="docs", query=vector,
        limit=3, score_threshold=0.3
    )
    if not results.points:
        return "No relevant information found in the knowledge base."
    return "\n".join([r.payload["text"] for r in results.points])

AVAILABLE = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
    "search_knowledge_base": search_knowledge_base,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current date and time",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate a basic math expression like 2+2 or 15*8",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "The math expression"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a location by latitude and longitude",
        "parameters": {"type": "object", "properties": {
            "latitude": {"type": "number"}, "longitude": {"type": "number"}},
            "required": ["latitude", "longitude"]}}},
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for information about Kubernetes, OCI, OKE, RAG, vector databases, FastAPI, and related cloud/AI topics",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query"}},
            "required": ["query"]}}},
]
