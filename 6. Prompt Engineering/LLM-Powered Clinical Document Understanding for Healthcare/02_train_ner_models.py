# 02_train_ner_models.py
# 2.1 Import Necessary Libraries
import pandas as pd
import argparse
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report

# 2.
# 2.2 Model Registry

MODEL_REGISTRY = {
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biomed_roberta": "allenai/biomed_roberta_base"
}


# 2.3 Preprocessing

def load_and_format_data(path):
    df = pd.read_csv(path)
    texts = df['text'].unique()

    formatted = []
    for text in texts:
        sub_df = df[df['text'] == text]
        entities = [(row['start'], row['end'], row['tag_name']) for _, row in sub_df.iterrows()]
        formatted.append({'text': text, 'entities': entities})
    return formatted

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128,
                                 return_offsets_mapping=True)
    labels = []

    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        word_labels = ["O"] * len(offsets)
        for start, end, tag in examples['entities'][i]:
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start >= start and tok_end <= end:
                    prefix = "B" if word_labels[idx] == "O" else "I"
                    word_labels[idx] = f"{prefix}-{tag}"
        labels.append([label2id.get(lbl, 0) for lbl in word_labels])

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 2.4 Training Function

def train_model(model_key):
    print(f"[INFO] Training model: {model_key}")
    model_name = MODEL_REGISTRY[model_key]

    # Load and format data
    formatted = load_and_format_data('./data/Corona2_annotations.csv')
    dataset = Dataset.from_list(formatted)

    # Extract unique labels
    all_tags = set(tag for ex in formatted for _, _, tag in ex['entities'])
    label_list = ['O'] + [f"B-{tag}" for tag in all_tags] + [f"I-{tag}" for tag in all_tags]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Tokenize + align labels
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=True)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    training_args = TrainingArguments(
        output_dir=f"./models/{model_key}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_total_limit=1,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"./models/{model_key}")
    tokenizer.save_pretrained(f"./models/{model_key}")
    print(f"[INFO] Model saved to ./models/{model_key}")


# 2.5 CLI Interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model on clinical data.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True, help="Which model to train.")
    args = parser.parse_args()

    os.makedirs('./models', exist_ok=True)
    train_model(args.model)
'''
python 02_train_ner_models.py --model biobert

# Train ClinicalBERT
python 02_train_ner_models.py --model clinicalbert

# Train BioMed-RoBERTa
python 02_train_ner_models.py --model biomed_roberta
'''