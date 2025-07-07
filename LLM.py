from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_ai_description(detected_objects, relationships):
    labels = list({obj['label'] for obj in detected_objects})

    prompt = f"The following objects are detected: {', '.join(labels)}.\n"
    prompt += "Their spatial relationships are as follows:\n"
    for rel in relationships:
        prompt += f"- {rel}\n"
    prompt += "Generate a natural description of this scene for a visually impaired person."

    encoded = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            max_length=100
        )

    description = tokenizer.decode(output[0], skip_special_tokens=True)
    return description[len(prompt):].strip()

