import argparse
import os
from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Label mapping
label2id = {
    'O': 0,
    'DIAGNOSIS': 1,
    'MEDICATION': 2,
    'SYMPTOM': 3,
    'TREATMENT': 4,
    'MEDICINE': 5,
    'MEDICALCONDITION': 6,
    'PATHOGEN': 7,
    # Add more labels if needed
}
id2label = {v: k for k, v in label2id.items()}


def load_and_prepare_data():
    df = pd.read_csv('./data/Corona2_annotations.csv')

    # Group by text (document), each text may have multiple annotations
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row['text']].append({
            'start': int(row['start']),
            'end': int(row['end']),
            'label': row['tag_name']
        })

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

    samples = []

    for text, entities in grouped.items():
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding='max_length',
            max_length=512
        )

        labels = ['O'] * len(encoding['input_ids'])

        offset_mapping = encoding['offset_mapping']

        for entity in entities:
            start_char = entity['start']
            end_char = entity['end']
            tag = entity['label']

            for idx, (start, end) in enumerate(offset_mapping):
                if start is None or end is None:
                    continue
                if start >= end_char:
                    break
                if end > start_char and start < end_char:
                    labels[idx] = tag

        label_ids = [label2id.get(label, 0) for label in labels]

        encoding['labels'] = label_ids
        del encoding['offset_mapping']  # Remove offset mapping, not needed for training

        samples.append(encoding)

    dataset = Dataset.from_list(samples)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    return dataset['train'], dataset['test'], tokenizer


def train_model(model_name):
    train_dataset, val_dataset, tokenizer = load_and_prepare_data()

    if model_name == "biobert":
        model = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=len(label2id), id2label=id2label, label2id=label2id)
    elif model_name == "clinicalbert":
        model = AutoModelForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=len(label2id), id2label=id2label, label2id=label2id)
    elif model_name == "biomed_roberta":
        model = AutoModelForTokenClassification.from_pretrained('Rostlab/prot_bert_bfd', num_labels=len(label2id), id2label=id2label, label2id=label2id)
    else:
        raise ValueError(f"Model {model_name} is not supported")

    training_args = TrainingArguments(
        output_dir=f'./results/{model_name}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(f'./saved_models/{model_name}')
    tokenizer.save_pretrained(f'./saved_models/{model_name}')

    print(f"Model {model_name} trained and saved successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['biobert', 'clinicalbert', 'biomed_roberta'], help="Model to train")
    args = parser.parse_args()

    print(f"[INFO] Training model: {args.model}")
    train_model(args.model)


if __name__ == '__main__':
    main()
