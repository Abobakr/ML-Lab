from transformers import pipeline


generator = pipeline("text-generation", model="distilgpt2")
result = generator(
	"Machine learning is",
	max_new_tokens=30,
	num_return_sequences=1,
	do_sample=False,
	clean_up_tokenization_spaces=False,
)

print(result[0]["generated_text"])
