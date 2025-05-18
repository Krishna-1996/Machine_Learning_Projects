#%%
# Step 1:Load and Preprocess the dataset
# 1.1 The Corona2.json Data

import json
import pandas as pd

# Load the JSON data from the 'Corona2.json' file
with open('Corona2.json', 'r') as file:
    data = json.load(file)

# Verify the first few records to understand the structure
print(data['examples'][0].keys())  # Check top-level keys

#%%
# 1.2 The CSV Output Data
def flatten_json(data):
    records = []
    for example in data['examples']:
        text = example['content']  # This is the unstructured text content
        annotations = example['annotations']  # List of annotations (entities)
        
        # For each annotation, we create a row of data
        for annotation in annotations:
            record = {
                'text': text,
                'start': annotation['start'],
                'end': annotation['end'],
                'tag_name': annotation['tag_name'].upper()  # Normalize tag names to uppercase
            }
            records.append(record)
    
    return pd.DataFrame(records)

# Flatten data into a dataframe and save as CSV
df = flatten_json(data)
df.to_csv('Corona2_annotations.csv', index=False)

print("CSV file created: 'Corona2_annotations.csv'")

# %%
# Step 2: Preprocess the Data for LLM-based Prompt Extraction
def prepare_data_for_llm(data):
    training_data = []
    for example in data['examples']:
        text = example['content']
        entities = [
            (annotation['start'], annotation['end'], annotation['tag_name'].upper())
            for annotation in example['annotations']
        ]
        training_data.append({
            'text': text,
            'entities': entities
        })
    return training_data

# Prepare data for LLM extraction (JSON format)
llm_data = prepare_data_for_llm(data)

# Save the LLM data to a new JSON file
with open('Corona2_for_llm.json', 'w') as outfile:
    json.dump(llm_data, outfile)

print("LLM data JSON file created: 'Corona2_for_llm.json'")



# %%
# Step 3: Implement OpenAI LLM Extraction
# pip install openai

# 3.2 Extract Entities Using OpenAI GPT
import openai

# Initialize OpenAI API client (replace with your own API key)
openai.api_key = "YOUR_OPENAI_API_KEY"

def extract_entities_with_openai(text):
    # Define the prompt structure for extracting entities
    prompt = f"""
    Extract the following entities from the clinical text: diagnoses, medications, and symptoms.

    Text: {text}

    Output should be in the following format:
    {{
        "diagnoses": ["<diagnosis_1>", "<diagnosis_2>"],
        "medications": ["<medication_1>", "<medication_2>"],
        "symptoms": ["<symptom_1>", "<symptom_2>"]
    }}
    """

    response = openai.Completion.create(
        engine="text-davinci-003",  # Use the appropriate GPT model
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    return response.choices[0].text.strip()

# Example: Run extraction on the first entry from the JSON data
text_example = llm_data[0]['text']
extracted_entities = extract_entities_with_openai(text_example)
print(extracted_entities)


# %%
# Step 4: Train BERT for Named Entity Recognition (NER)

# 4.1 4.1 Install the Required Libraries

# pip install transformers datasets

# 4.2 Convert CSV Data into a Format BERT Can Use
from datasets import Dataset
import pandas as pd

# Load the CSV file
df = pd.read_csv('Corona2_annotations.csv')

# Example conversion: Create tokenized input for BERT
def prepare_bert_data(df):
    texts = []
    labels = []
    for _, row in df.iterrows():
        text = row['text']
        start = row['start']
        end = row['end']
        tag = row['tag_name']
        
        # Tokenize text and create labels (one token = one label)
        tokens = text.split()
        label = ['O'] * len(tokens)  # "O" for non-entity tokens
        
        # Label the entity tokens
        entity_tokens = tokens[start:end+1]
        for i in range(start, end+1):
            label[i] = tag  # Assign the correct entity label

        texts.append(tokens)
        labels.append(label)
    
    return texts, labels

texts, labels = prepare_bert_data(df)

# Convert to HuggingFace's dataset format
dataset = Dataset.from_dict({
    'tokens': texts,
    'labels': labels
})

print(dataset)

# %%
# 4.3 Fine-tune BERT on the Data
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments

# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['tokens'], padding=True, truncation=True)

dataset = dataset.map(tokenize_function, batched=True)

# Load BERT for Token Classification
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(set(df['tag_name'])))

# Set up Trainer
training_args = TrainingArguments(
    output_dir='./results',  
    evaluation_strategy="epoch",  
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

# Train the model
trainer.train()



# %%
# Step 5: Evaluate the Models
from sklearn.metrics import classification_report

# Example of evaluation (you will need to provide a separate test dataset)
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=-1)

print(classification_report(test_labels, predicted_labels))


# %%
# 5.2 Evaluate OpenAI LLM-based Extraction
def evaluate_llm_extraction(extracted, ground_truth):
    # Compare extracted entities with ground truth entities
    pass  # Implement comparison logic here (precision, recall, F1)



# %%




# %%



