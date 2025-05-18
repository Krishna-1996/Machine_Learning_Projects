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
# Step 3.2:
print("id : Examples: ")
data['id'][10]['examples'][10]

print("Examples : content: ")
data['examples'][10]['content'][10]

print("Examples : metadata: ")
data['examples'][10]['metadata'][10]

print("Examples : annotations: ")
data['examples'][10]['annotations'][10]

print("Examples : classifications: ")
data['examples'][10]['classifications'][10]

# %%
# Step 6:



# %%
# Step 7:

# %%
# Step 


# %%
# Step 


# %%
# Step 

# %%
# Step 


# %%
# Step 