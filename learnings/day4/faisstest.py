import faiss


from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")

docs=["how to reset your zomato password",
 "track your zomato delivery live",
 "cancel a zomato order before pickup",
 "Zomato gold membership benifits",
 "contact zomato customer support",]

doc_embeddings=model.encode(docs).astype('float32')

index=faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

query="I Want to cancel my food order"
q_emb=model.encode([query]).astype('float32')

distances,indices=index.search(q_emb, k=2)
for i,idx in enumerate(indices[0]):
    print(f"Result {i+1}: {docs[idx]} (score: {distances[0][i]:.2f})")
