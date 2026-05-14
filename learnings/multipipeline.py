from transformers import pipeline
sa =  pipeline("sentiment-analysis")
print (sa("This food quality is excellent!"))

ner = pipeline("ner", aggregation_strategy="simple")
print(ner("Infosys  based in Bengaluru, Karnataka"))

qa = pipeline("question-answering")
context = "python was created by Guido van Rossum and released in 1991. It is widely used in AI"
summ = pipeline("summarization")
long_text = "Generative AI refers to AI systems that can generate content such as text, images, or music. These models learn patterns from large datasets and can create new content that resembles the training data. Generative AI has applications in various fields, including art, entertainment, and natural language processing." 
print(summ(long_text, max_length=60, min_length=20))
