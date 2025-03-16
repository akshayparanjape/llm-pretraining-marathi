from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "models/marathi-llm"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Sample Marathi text for testing
marathi_text = "प्राचीन काळापासून महाराष्ट्रात"

# Generate text
input_ids = tokenizer(marathi_text, return_tensors="pt").input_ids
generated_outputs = model.generate(
    input_ids, 
    max_length=100, 
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
)

# Decode and print the generated text
generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
print(generated_text)
