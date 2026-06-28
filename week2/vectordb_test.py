from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")

client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
print("Collection created successfully")

docs = [
    "Kubernetes is an open-source container orchestration system",
    "A pod is the smallest deployable unit in Kubernetes",
    "Helm is a package manager for Kubernetes applications",
    "OCI offers managed Kubernetes through Oracle Container Engine",
    "Vector databases store embeddings for semantic search",
    "RAG combines retrieval with language model generation",
    "FastAPI is a modern Python web framework for building APIs",
    "Microservices split applications into small independent services",
]

points = [
    PointStruct(
        id=i,
        vector=model.encode(doc).tolist(),
        payload={"text": doc}
    )
    for i, doc in enumerate(docs)
]

client.upsert(collection_name="docs", points=points)
print(f"Indexed {len(points)} documents")

def semantic_search(query: str, limit: int = 3, min_score: float = 0.3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name="docs",
        query=query_vector,
        limit=limit,
        score_threshold=min_score
    )
    return results.points

# test with varied queries
queries = [
    "how do I deploy apps on k8s?",
    "package management for kubernetes",
    "how does semantic search work?",
    "python web framework",
    "what is RAG?",
    "I want to go for a swim",
]

for q in queries:
    print(f"\nQuery: {q}")
    results = semantic_search(q)
    if results:
        for r in results:
            print(f"  {r.score:.3f} | {r.payload['text']}")
    else:
        print("  no results above threshold")
