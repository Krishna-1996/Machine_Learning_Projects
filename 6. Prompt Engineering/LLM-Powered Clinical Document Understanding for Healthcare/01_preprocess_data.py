# 01_preprocess_data.py

import json
import pandas as pd

# Load the Corona2.json dataset
with open("Corona2.json", "r") as f:
    data = json.load(f)

# Save flattened annotations for BERT training
def flatten_json(data):
    rows = []
    for example in data["examples"]:
        text = example["content"]
        for ann in example["annotations"]:
            rows.append({
                "text": text,
                "start
