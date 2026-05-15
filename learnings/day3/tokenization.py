from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Zomato delivers food"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("Token IDs:", ids)

decoded_text = tokenizer.decode(ids)
print("Decoded Text:", decoded_text)
