# AI-SPARTENS
PROJECT
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="ibm-granite/granite-guardian-3.2-5b-lora-harm-correction")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-guardian-3.2-5b-lora-harm-correction")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-guardian-3.2-5b-lora-harm-correction")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
