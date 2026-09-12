import matplotlib.pyplot as plt
from pathlib import Path
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
prompt = "Machine learning is"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
	**inputs,
	max_new_tokens=15,
	do_sample=False,
	return_dict_in_generate=True,
	output_scores=True,
)

generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
tokens = []
probabilities = []
for score, token_id in zip(output.scores, generated_ids):
	token = tokenizer.decode([token_id])
	if "\n" in token:
		break
	tokens.append(token)
	probabilities.append(torch.softmax(score[0], dim=-1)[token_id].item())
full_text = tokenizer.decode(output.sequences[0][:inputs["input_ids"].shape[1] + len(tokens)]).strip()
print(full_text)

# Visualization: chosen-token probabilities show the model's confidence at each step.
figure, axis = plt.subplots(figsize=(10, 4))
axis.bar(range(len(tokens)), probabilities, color="steelblue")
axis.set(
	title="Token-by-token generation confidence",
	xlabel="Generated token",
	ylabel="Probability of chosen token",
	xticks=range(len(tokens)),
)
axis.set_xticklabels(tokens, rotation=45, ha="right")
axis.set_ylim(0, 1)
figure.tight_layout()
Path("outputs").mkdir(exist_ok=True)
figure.savefig(f"outputs/{Path(__file__).parent.name}_{Path(__file__).stem}.png", dpi=150)
