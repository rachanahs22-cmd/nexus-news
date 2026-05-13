from transformers import pipeline
gen=pipeline("text-generation", model="gpt2")
result=gen("The future of AI in india is",max_new_token=50,num_return_sequences=1)
print(result[0]['generated_text'])
creative=gen("Once upon a time in mysuru",max_new_token=60,temperature=1.2,do_sample=True)
focused=gen("Once upon a time in mysuru",max_new_token=60,temperature=0.3,do_sample=True)

print("Creative:",creative[0]['generated_text'])
print("Focused:",focused[0]['generated_text'])