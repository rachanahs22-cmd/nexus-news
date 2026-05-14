import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

docs = ["How to reset your Zomato passwors",
    "Track your Zomato delivery live",
    "Cancel a Zomato Order before pickup",
    "Zomato Gold Membership Benefits",
    "Contact Zomato Customer Support"]

docs_embeddings = model.encode(docs).astype('float32')
index = faiss.IndexFlatL2(docs_embeddings.shape[1])
index.add(docs_embeddings)

query = "I want to cancel my order"
query_embedding = model.encode([query]).astype('float32')

distances, indices = index.search(query_embedding, k=3)
for i, idx in enumerate(indices[0]):
    print(f"Result {i+1}: {docs[idx]} (Score: {distances[0][i]:.4f})")
    