from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
text="Zomato delivers food"
tokens=tokenizer.tokenize(text)
ids=tokenizer.encode(text)

print("Tokens:",tokens)
print("IDs:",ids)

decoded=tokenizer.decode(ids)
print("Decoded:",decoded)