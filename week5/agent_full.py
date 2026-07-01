import ollama
import httpx
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- set up the knowledge base (same as week 3) ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(":memory:")
qdrant.create_collection(
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
print(f"Indexed {len(points)} chunks\n")

# --- tools ---
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

# --- agent loop ---
def run_agent(question: str, model="llama3.1:8b", max_steps=5):
    messages = [{"role": "user", "content": question}]
    for step in range(max_steps):
        response = ollama.chat(model=model, messages=messages, tools=TOOLS)
        msg = response["message"]
        messages.append(msg)

        if not msg.tool_calls:
            return msg["content"]

        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments
            print(f"  [step {step+1}] calling {name}({args})")
            result = AVAILABLE[name](**args)
            print(f"            -> {str(result)[:80]}")
            messages.append({"role": "tool", "content": str(result), "tool_name": name})

    return "Max steps reached"

# --- test it ---
if __name__ == "__main__":
    questions = [
        "What is OKE?",
        "What time is it and what is 145 * 23?",
        "What's the weather in Raleigh? Latitude 35.78, longitude -78.64",
    ]
    for q in questions:
        print(f"Q: {q}")
        answer = run_agent(q)
        print(f"A: {answer}\n")
