# 1. Import necessary Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras 
import tensorflow as tf 

# 2. Import Dataset
ipl = pd.read_csv('E:/Machine Learning/Machine Learning/Machine_Learning_Projects/Machine_Learning_Projects/1. IPL Score Prediction/ipl_data.csv')
'''print(ipl.head())'''

# 3. Data Pre-Processing
# 3.1 Drop Unnecessary Columns
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis =1)
# 3.2 Define the x(independent variable) and y(depended variable)
x =  df.drop(['total'], axis =1)
y = df['total']
print(y)