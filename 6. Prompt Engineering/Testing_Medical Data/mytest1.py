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
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %%
# Step 3: Read and check Dataset
data = pd.read_json('/kaggle/input/medical-ner/Corona2.json')
data.head()


# %%
# Step 4:
list(data['examples'][0].keys())


# %%
# Step 5:
data['examples'][0]['content']


# %%
# Step 6:
data['examples'][0]['annotations'][0]


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