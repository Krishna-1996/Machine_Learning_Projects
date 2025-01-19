# main_project.py (Training multiple models)

import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical

BASE_DATA_DIR = r"D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1"
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, "processed_data")

X_data = []
y_data = []

# Load features and labels
for idx in range(100):
    feature_file = os.path.join(PROCESSED_DATA_DIR, f'features_{idx}.npy')
    if os.path.exists(feature_file):
        features = np.load(feature_file)
        X_data.append(features)
        
        label_file = os.path.join(BASE_DATA_DIR, 'labels.pkl')
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)
        
        label = labels[idx]
        y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Function to create and save a model
def create_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], )))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and save 10 models
for i in range(10):
    model = create_model()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler])
    
    # Save each model to a unique file
    model.save(f'D:/Machine_Learning_Projects/5. Real-Time Deepfake Detection in Video Streams project/my_webpage/models/model_{i}.h5')

