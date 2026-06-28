from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")

def load_and_chunk(filepath: str):
    with open(filepath, "r") as f:
        content = f.read()
    chunks = [
        c.strip() for c in content.split("\n\n")
        if c.strip() and not c.startswith("#")
    ]
    return chunks

def index_documents(chunks, collection="docs"):
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=384, distance=Distance.COSINE
        )
    )
    embeddings = model.encode(chunks)
    points = [
        PointStruct(
            id=i,
            vector=emb.tolist(),
            payload={
                "text": chunk,
                "source": "docs.txt",
                "chunk_index": i
            }
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=collection, points=points)
    print(f"Indexed {len(points)} chunks")
    return client

def semantic_search(client, query: str, limit: int = 3, min_score: float = 0.3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name="docs",
        query=query_vector,
        limit=limit,
        score_threshold=min_score
    )
    return results.points

if __name__ == "__main__":
    chunks = load_and_chunk("docs.txt")
    print(f"Loaded {len(chunks)} chunks from docs.txt")
    for i, c in enumerate(chunks[:3]):
        print(f"\nChunk {i}: {c[:80]}...")

    client = index_documents(chunks)

    print("\n--- semantic search test ---")
    queries = [
        "how does OKE work?",
        "what is RAG?",
        "how do I store secrets securely?",
        "kubernetes deployment scaling",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = semantic_search(client, q)
        if results:
            for r in results:
                print(f"  {r.score:.3f} | {r.payload['text'][:80]}...")
        else:
            print("  no results above threshold")