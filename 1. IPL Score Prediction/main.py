# 1. IMPORT NECESSARY LIBRARIES:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras 
import tensorflow as tf 

# 2. IMPORT DATASET
ipl = pd.read_csv('E:/Machine Learning/Machine Learning/Machine_Learning_Projects/Machine_Learning_Projects/1. IPL Score Prediction/ipl_data.csv')
'''print(ipl.head())'''

# 3. DATA PRE-PROCESSING
# 3.1 Drop Unnecessary Columns
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis =1)

# 3.2 Define the x(independent variable) and y(depended variable)
x =  df.drop(['total'], axis =1)
y = df['total']
'''print(y)'''

# 3.3 Label Encoding
from sklearn.preprocessing import LabelEncoder
# Create a label encoder object for each categorical features.
venue_encoder = LabelEncoder()	
bat_team_encoder = LabelEncoder()	
bowl_team_encoder = LabelEncoder()	
batsman_encoder = LabelEncoder()	
bowler_encoder = LabelEncoder()
# Fit & Transform the categorical features with Label Encoding
x['venue'] = venue_encoder.fit_transform(x['venue'])
x['bat_team'] = bat_team_encoder.fit_transform(x['bat_team'])
x['bowl_team'] = bowl_team_encoder.fit_transform(x['bowl_team'])
x['batsman'] = batsman_encoder.fit_transform(x['batsman'])
x['bowler'] = bowler_encoder.fit_transform(x['bowler'])

# 3.4 Train_Test_Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 3.5 Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler on the training data and transform both
# training and testing data.
x_train_scalar = scaler.fit_transform(x_train)
x_test_scaler = scaler.fit_transform(x_test)

# !!!
# 4. DEFINE THE NEURAL NETWORK
model = keras.Sequential([
    keras.layers.Input(shape=(x_train_scalar))


])








