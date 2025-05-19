# 04_infer_and_compare.py
# 4.1 Import necessary Libraries
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

# 4.2 Models to Load

MODEL_PATHS = {
    "BioBERT": "./models/biobert",
    "ClinicalBERT": "./models/clinicalbert",
    "BioMed-RoBERTa": "./models/biomed_roberta",
}

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path)
    return tokenizer, model

def get_entities(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True)
    with torch.no_grad():
        outputs = model(**inputs).logits

    predictions = torch.argmax(outputs, dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    id2label = model.config.id2label
    offset_mapping = inputs["offset_mapping"].squeeze().tolist()

    entities = []
    current_entity = ""
    current_label = ""
    current_start = None

    for idx, (token, pred, offset) in enumerate(zip(tokens, predictions, offset_mapping)):
        label = id2label[pred]
        if label.startswith("B-"):
            if current_entity:
                entities.append((current_entity, current_label, current_start, offset[0]))
            current_entity = token
            current_label = label[2:]
            current_start = offset[0]
        elif label.startswith("I-") and current_label == label[2:]:
            current_entity += " " + token
        else:
            if current_entity:
                entities.append((current_entity, current_label, current_start, offset[0]))
                current_entity = ""
                current_label = ""
                current_start = None

    if current_entity:
        entities.append((current_entity, current_label, current_start, offset[0]))

    return entities

def main():
    sample_text = input("Enter a sample doctor's note:\n> ")

    print("\n--- Comparing NER Predictions ---\n")

    for model_name, path in MODEL_PATHS.items():
        print(f"🔍 {model_name} Prediction:")
        tokenizer, model = load_model_and_tokenizer(path)
        entities = get_entities(sample_text, tokenizer, model)

        if entities:
            for ent_text, ent_label, start, end in entities:
                print(f"  🟢 {ent_label}: \"{ent_text}\"")
        else:
            print("  ⚠️ No entities found.")
        print("-" * 40)

if __name__ == "__main__":
    main()
'''

🟢 What's Next?
You can now:

Add evaluation on test sets (metrics like F1, precision)

Visualize results or generate HTML reports

Optionally build a lightweight web UI using Streamlit or Flask

Would you like help building a basic UI or generating evaluation reports?

'''