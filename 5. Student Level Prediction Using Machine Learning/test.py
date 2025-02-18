import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load your dataset (assuming it is saved as a CSV file for simplicity)
data = pd.read_csv('your_dataset.csv')

# Preprocessing
# Encode categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'Current Year (17/18)', 'Proposed Year/Grade (18/19)',
                       'Year of Admission', 'Previous Curriculum (17/18)2', 'Current School',
                       'Current Curriculum', 'Previous year/Grade']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features and target
X = data.drop(['Math20-3 ', 'Science20-3 ', 'English20-3 '], axis=1)  # Drop target columns
Y = data[['Math20-3 ', 'Science20-3 ', 'English20-3 ']]  # Target columns

# Normalize numerical columns
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3)  # Output layer for 3 target columns
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Save the model
model.save('student_performance_model.h5')
