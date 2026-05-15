from transformers import pipeline
sa=pipeline("sentiment-analysis")
print(sa("Ithis food quality is excellent"))

ner=pipeline("ner",grouped_entities=True)
print(ner("infosys is based in banglore,karnataka."))

qa=pipeline("question-answering")
context="The capital of india is delhi"
summ=pipeline("summarization")
long_text="The history of India is rich and diverse, spanning thousands of years. It has been home to various civilizations, including the Indus Valley Civilization, the Maurya and Gupta Empires, and the Mughal Empire. India has also been influenced by various cultures and religions, such as Hinduism, Buddhism, Islam, and Christianity. The country has a vibrant cultural heritage, with a wide range of art, music, dance, and literature. Today, India is a rapidly developing nation with a growing economy and a diverse population."
print(summ(long_text, max_length=60, min_length=20))