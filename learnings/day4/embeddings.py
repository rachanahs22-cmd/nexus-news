from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["Zomato delivers food to your door", "Swiggy is a food delivery service,Python is a programming language"]

embeddings = model.encode(sentences)

print("Embeddings shape:", embeddings.shape)
print("First sentence embedding (First 5 values):", embeddings[0][:5])