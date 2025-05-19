import spacy
from spacy.training.example import Example
import random
import os

# Function to train a custom NER model
def train_ner_model(data_dir, output_model_dir, n_iter=10):
    # Load the preprocessed data
    nlp = spacy.blank('en')
    train_data = []

    # Load the training examples
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.spacy'):
            with open(os.path.join(data_dir, file_name), 'rb') as file:
                nlp.from_disk(data_dir)  # load preprocessed data

    # Create the NER pipeline if it doesn't exist
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    # Add labels to the NER pipe
    for text, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Split the data into train and dev sets
    random.shuffle(train_data)
    size = int(len(train_data) * 0.8)
    train_data, dev_data = train_data[:size], train_data[size:]

    # Training loop
    optimizer = nlp.begin_training()
    for epoch in range(n_iter):
        random.shuffle(train_data)
        losses = {}

        for batch in spacy.util.minibatch(train_data, size=8):
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses)
        
        print(f"Epoch {epoch} - Losses: {losses}")

    # Save the trained model to the output directory
    nlp.to_disk(output_model_dir)
    print(f"Model saved at {output_model_dir}")

# Example usage
if __name__ == "__main__":
    data_dir = 'path/to/preprocessed/data'
    output_model_dir = './model'
    train_ner_model(data_dir, output_model_dir, n_iter=10)
INPUT_JSON = './data/Corona2.json'
OUTPUT_CSV = './data/Corona2_annotations.csv'
OUTPUT_LLM_JSON = './data/Corona2_for_llm.json'