from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "Kubernetes orchestrates containerized workloads"
embedding = model.encode(sentence)

print(f"Type: {type(embedding)}")
print(f"Shape: {embedding.shape}")
print(f"First 5 values: {embedding[:5]}")

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

e1 = model.encode("Kubernetes manages containers")
e2 = model.encode("K8s orchestrates containerized apps")
e3 = model.encode("I enjoy cooking curries on weekends")

print(f"\nSimilar pair:   {cosine_similarity(e1, e2):.4f}")
print(f"Different pair: {cosine_similarity(e1, e3):.4f}")