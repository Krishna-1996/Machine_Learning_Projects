# 01_preprocess_data.py

import json
import pandas as pd
import os

# File paths
INPUT_JSON = './data/Corona2.json'
OUTPUT_CSV = './data/Corona2_annotations.csv'
OUTPUT_LLM_JSON = './data/Corona2_for_llm.json'

def flatten_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    records = []
    llm_format = []

    for example in data['examples']:
        text = example['content']
        annotations = example.get('annotations', [])

        # For LLM-style output
        entities = []

        for ann in annotations:
            start, end, tag = ann['start'], ann['end'], ann['tag_name'].upper()
            records.append({
                'text': text,
                'start': start,
                'end': end,
                'tag_name': tag
            })
            entities.append((start, end, tag))

        llm_format.append({
            'text': text,
            'entities': entities
        })

    df = pd.DataFrame(records)
    return df, llm_format

def main():
    print("[INFO] Preprocessing started...")
    os.makedirs('./data', exist_ok=True)

    df, llm_data = flatten_json(INPUT_JSON)
    
    # Save CSV for BERT models
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Annotations saved to {OUTPUT_CSV}")

    # Save JSON for LLM-based rule extraction
    with open(OUTPUT_LLM_JSON, 'w') as f:
        json.dump(llm_data, f, indent=2)
    print(f"[INFO] LLM data saved to {OUTPUT_LLM_JSON}")

if __name__ == '__main__':
    main()
''' 

To run this :# Train BioBERT
python 02_train_ner_models.py --model biobert

# Train ClinicalBERT
python 02_train_ner_models.py --model clinicalbert

# Train BioMed-RoBERTa
python 02_train_ner_models.py --model biomed_roberta

'''