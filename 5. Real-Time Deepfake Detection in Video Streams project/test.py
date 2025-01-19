# %% Importing necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image

# %% Load your dataset
# Assuming you already have a DataFrame with paths and labels, here is an example:
# df = pd.read_csv("your_data.csv")  # Your CSV file containing file paths and labels

# Example class distribution for fake/real
# df['Label'] = ['fake' or 'real']

# %% Preprocess the data (load frames, resize, normalize)
def extract_frames_from_video(video_path, num_frames=30):
    # Placeholder function to simulate frame extraction from video
    # Replace this with actual code to extract frames from video files
    frames = np.random.rand(num_frames, 224, 224, 3)  # Example: 30 frames, 224x224 RGB
    return frames

# %% Prepare the dataset for model input
X_data = []
y_data = []

# Loop through the DataFrame and extract frames for each video
for index, row in df.iterrows():
    video_path = row['video_path']  # Adjust the column name if needed
    label = row['Label']  # Ensure labels are 0 for fake, 1 for real
    frames = extract_frames_from_video(video_path)
    X_data.append(frames)
    y_data.append(0 if label == 'fake' else 1)

X_data = np.array(X_data)
y_data = np.array(y_data)

# %% Reshape and split data into train/test sets
# X_data shape should be (num_samples, 30, 224, 224, 3)
# Flatten frames for LSTM (time_steps, features)
X_data_reshaped = X_data.reshape(X_data.shape[0], X_data.shape[1], -1)  # (num_samples, 30, 224*224*3)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_data_reshaped, y_data, test_size=0.2, random_state=42)

# %% One-hot encoding for labels
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

# %% Compute class weights for imbalanced dataset
class_counts = pd.Series(y_data).value_counts()
class_weights = {0: len(y_data) / (2 * class_counts[0]), 1: len(y_data) / (2 * class_counts[1])}

# %% Build the Model
model = Sequential()

# LSTM layer
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # 30 frames, flattened 224*224*3 features
model.add(LSTM(256, return_sequences=False))  # LSTM layer
model.add(Dropout(0.3))  # Dropout to avoid overfitting
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # L2 regularization
model.add(Dense(2, activation='softmax'))  # Output layer for binary classification

# %% Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# %% Set up callbacks for early stopping and learning rate scheduling
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lambda epoch: 0.0005 * 0.9 ** epoch)

# %% Train the model
print("Training the model with enhancements...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stopping, lr_scheduler]
)

# %% Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# %% Model summary
model.summary()

# %% Save the model if desired
model.save('deepfake_detection_model.h5')
