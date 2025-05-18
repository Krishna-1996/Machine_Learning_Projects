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