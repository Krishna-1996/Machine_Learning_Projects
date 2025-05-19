import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# Function to encode labels (from strings to integers)
def encode_labels(tags, label2id):
    return [label2id.get(tag, -1) for tag in tags]

# Define the label mapping (string to integer)
label2id = {
    'O': 0,  # Non-entity token
    'DIAGNOSIS': 1,
    'MEDICATION': 2,
    'SYMPTOM': 3,
    'TREATMENT': 4,
    # Add other labels here as needed
}

# Function to load dataset and prepare it for training
def load_and_prepare_data():
    # Load your dataset (assumes a CSV file; you can adjust this as needed)
    # Replace 'Corona2_annotations.csv' with your actual dataset file
    dataset = load_dataset("csv", data_files="./data/Corona2_annotations.csv", delimiter=",")
    
    # Process dataset and prepare it in the required format (tokens and labels)
    def preprocess_data(examples):
        texts = examples['text']
        labels = examples['tag_name']
        
        # Convert text to tokens and labels to integers
        tokenized_inputs = tokenizer(texts, padding=True, truncation=True, is_split_into_words=True)
        label_ids = [encode_labels(label.split(), label2id) for label in labels]  # Split labels by spaces
        
        # Add labels to tokenized inputs
        tokenized_inputs['labels'] = label_ids
        return tokenized_inputs
    
    # Tokenizer: change based on your model (BioBERT, ClinicalBERT, BioMed-RoBERTa)
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')  # Change based on the model
    
    # Preprocess the dataset
    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    
    # Split into train and validation sets
    train_dataset, val_dataset = train_test_split(tokenized_dataset['train'], test_size=0.1, random_state=42)
    
    return train_dataset, val_dataset, tokenizer

# Function to train the model
def train_model(model_name):
    # Load the dataset
    train_dataset, val_dataset, tokenizer = load_and_prepare_data()
    
    # Load the model based on the specified model
    if model_name == "biobert":
        model = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=len(label2id))
    elif model_name == "clinicalbert":
        model = AutoModelForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=len(label2id))
    elif model_name == "biomed_roberta":
        model = AutoModelForTokenClassification.from_pretrained('Rostlab/prot_bert_bfd', num_labels=len(label2id))
    else:
        raise ValueError(f"Model {model_name} is not supported")
    
    # Prepare the training arguments
    training_args = TrainingArguments(
        output_dir=f'./results/{model_name}',  # Where the model checkpoints will be saved
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",  # Evaluate after each epoch
        save_strategy="epoch",  # Save the model after each epoch
        load_best_model_at_end=True,  # Load the best model when finished training
    )

    # Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model after training
    model.save_pretrained(f'./saved_models/{model_name}')
    tokenizer.save_pretrained(f'./saved_models/{model_name}')

    print(f"Model {model_name} trained and saved successfully!")

# Main function to parse arguments and run the training process
def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['biobert', 'clinicalbert', 'biomed_roberta'], help="Model to train")
    args = parser.parse_args()
    
    # Call the function to train the model based on the input argument
    print(f"[INFO] Training model: {args.model}")
    train_model(args.model)

if __name__ == '__main__':
    main()

# 1. Train BioBERT:
# Run this in terminal: python 02_train_ner_models.py --model biobert

# 2. Train ClinicalBERT:
# Run this in terminal: python 02_train_ner_models.py --model clinicalbert

# 3. Train BioMed-RoBERTa:
# Run this in terminal: python 02_train_ner_models.py --model biomed_roberta
