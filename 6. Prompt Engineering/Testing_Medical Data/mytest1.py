# %%
# Step 1: Import necessary libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

# %%
# Step 2: Load Dataset
data_path = './Corona2.json'  # or just 'Corona2.json' if you're already in the script's directory
data = pd.read_json(data_path)
data.head()

# %%
# Step 3: Check Columns name and values
list(data['examples'][0].keys())

# %%
# Step 3.1: Check content with example column
data['examples'][0]['content']

# %%
# Inspect the top-level columns of the DataFrame
print(data.columns)

# Check the first few rows of the DataFrame
print(data.head())

# %%
# Step 3.2:
print("Examples : annotations: ")
data['examples'][0]['annotations'][0]


# %%
# Step 4: Data Transform
training_data = [{'text': example['content'],
                  'entities': [(annotation['start'], annotation['end'], annotation['tag_name'].upper())
                               for annotation in example['annotations']]}
                 for example in data['examples']]


# Step 4.1:
training_data[0]['entities']

# %%
training_data[0]['text'][563:571]
# %%
# Step :5 Processing text data using the SpaCy
nlp = spacy.blank("en") 
doc_bin = DocBin()
# %%
# Step 6: Processing training data by tokenizing text
from spacy.util import filter_spans

for training_example in tqdm(training_data):
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.set_ents(filtered_ents)
    doc_bin.add(doc)

doc_bin.to_disk("train.spacy")


# %%
# Step 7: SpaCy configuration (NER)
! python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency

# %%
# Step 8: Model training using SpaCy
! python -m spacy train config.cfg --output ./ --paths.train ./train.spacy --paths.dev ./train.spacy
# %%
# Step 9: 
nlp_trained_model = spacy.load("model-best")

# %%
# Step 10: Model Testing
doc = nlp_trained_model('''
The patient was prescribed Aspirin for their heart condition.
The doctor recommended Ibuprofen to alleviate the patient's headache.
The patient is suffering from diabetes, and they need to take Metformin regularly.
After the surgery, the patient experienced some post-operative complications, including infection.
The patient is currently on a regimen of Lisinopril to manage their high blood pressure.
The antibiotic course for treating the bacterial infection should be completed as prescribed.
The patient's insulin dosage needs to be adjusted to better control their blood sugar levels.
The physician suspects that the patient may have pneumonia and has ordered a chest X-ray.
The patient's cholesterol levels are high, and they have been advised to take Atorvastatin.
The allergy to penicillin was noted in the patient's medical history.
''')

# %%
# Step 11: Conclusion
spacy.displacy.render(doc, style="ent", jupyter=True)



 
# %%
# Step 1:



 
# %%
# Step 1:



 
# %%
# Step 1:



 