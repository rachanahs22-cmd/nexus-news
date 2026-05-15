from transformers import pipeline
gen =pipeline("text-generation", model="gpt2")
result = gen("The future of AI in Indian is", max_new_tokens=50, num_return_sequences=1)
print(result[0]['generated_text'])
creative = gen("Once upon a time in Mysuru", max_new_tokens=60,temperature=1.2, do_sample=True)
focused =gen("Once upon a time in Mysuru", max_new_tokens=60,temperature=0.3, do_sample=True)

print("Creative:", creative[0]['generated_text'])
print("Focused:", focused[0]['generated_text'])
