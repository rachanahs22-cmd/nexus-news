from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('all-MiniLM-L6-v2')
s1 = ["Zomato delivers food to your door"]
s2 = ["Swiggy is a food delivery service"]
s3 = ["Python is a programming language"]

e1 = model.encode(s1)
e2 = model.encode(s2)
e3 = model.encode(s3)

sim1 = util.cos_sim(e1,e2) # Similarity between s1 and s2
sim2 = util.cos_sim(e1,e3) # Similarity between s1 and s3
print(f"Zomato Vs Swiggy: {sim1.item():.3f}")
print(f"Zomato Vs Python: {sim2.item():.3f}")
