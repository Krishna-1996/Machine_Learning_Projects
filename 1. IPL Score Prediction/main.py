import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. IMPORT NECESSARY LIBRARIES:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import keras 
import tensorflow as tf 

# 2. IMPORT DATASET
ipl = pd.read_csv('E:/Machine Learning/Machine Learning/Machine_Learning_Projects/Machine_Learning_Projects/1. IPL Score Prediction/ipl_data.csv')

# 3. DATA PRE-PROCESSING
# 3.1 Drop Unnecessary Columns
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis=1)

# 3.2 Define the x (independent variable) and y (dependent variable)
x =  df.drop(['total'], axis=1)
y = df['total']
#  added

# 3.3 Label Encoding
venue_encoder = LabelEncoder()	
bat_team_encoder = LabelEncoder()	
bowl_team_encoder = LabelEncoder()	
batsman_encoder = LabelEncoder()	
bowler_encoder = LabelEncoder()
# Fit_transform
x['venue'] = venue_encoder.fit_transform(x['venue'])
x['bat_team'] = bat_team_encoder.fit_transform(x['bat_team'])
x['bowl_team'] = bowl_team_encoder.fit_transform(x['bowl_team'])
x['batsman'] = batsman_encoder.fit_transform(x['batsman'])
x['bowler'] = bowler_encoder.fit_transform(x['bowler'])

# 3.4 Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 3.5 Feature Scaling
scaler = MinMaxScaler()
x_train_scalar = scaler.fit_transform(x_train)
x_test_scalar = scaler.transform(x_test)

# 4. DEFINE THE NEURAL NETWORK.
model = keras.Sequential([
    keras.layers.Input(shape=(x_train_scalar.shape[1],)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(216, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

# 5. MODEL TRAINING.
history = model.fit(x_train_scalar, y_train, epochs=50, batch_size=64, validation_data=(x_test_scalar, y_test))
# Store the training and validation loss values
model_losses = pd.DataFrame(history.history)
# Plot training and validation 
model_losses.plot()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# 6. MODEL EVALUATION
predict = model.predict(x_test_scalar)
# MSE and MAE
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, predict)
mse = mean_squared_error(y_test, predict)
print("Mean Absolute Error: ", mae)
print("Mean Square Error: ", mse)
