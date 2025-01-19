
# %%
# Import necessary modules and libraries
import os
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# %%
# Path to main directory and CSV file
base_dir = r"D:\MSc. Project DeepFake Detection Datasets\Celeb-DF-v1"
csv_file = os.path.join(base_dir, "Video_Label_and_Dataset_List.csv")

# Directory where the processed features are saved
processed_data_dir = os.path.join(base_dir, "processed_data")

# Load CSV file
df = pd.read_csv(csv_file)

# Load pre-extracted features and labels
X_data = []
y_data = []

# %%
# Load features and labels from the saved files
for idx, row in df.iterrows():
    video_path = row['Video Path']
    label = 1 if row['Label'] == 'fake' else 0  # Fake -> 1, Real -> 0
    
    # Create the file path to the extracted features
    feature_file = os.path.join(processed_data_dir, f'features_{idx}.npy')
    
    # Check if the feature file exists
    if os.path.exists(feature_file):
        features = np.load(feature_file)
        X_data.append(features)
        y_data.append(label)

# Convert to NumPy arrays
X_data = np.array(X_data)
y_data = np.array(y_data)

# %%
# Ensure the data shapes are correct
print(f"Shape of X_data: {X_data.shape}")
print(f"Shape of y_data: {y_data.shape}")

# %%
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# %%
# One-hot encode the labels
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# %%
# Model architecture (Simple LSTM with dropout)
model = Sequential()
model.add(Input(shape=(X_train.shape[1], )))  # Each video has a flattened feature vector
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # Dense layer with L2 regularization
model.add(Dropout(0.3))  # Dropout layer to prevent overfitting
model.add(Dense(2, activation='softmax'))  # 2 output classes: real and fake

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# %%
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# %%
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Get detailed classification metrics
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# %%
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# %%
# Plot training and validation loss/accuracy curves
plt.figure(figsize=(12, 6))

# Training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
# %%
# Training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# %%
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights based on the class distribution
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train.argmax(axis=1))

# Create a dictionary for class weights
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Train the model with class weights
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,  # Use class weights here
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Get detailed classification metrics
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)


# %%
# Plot training and validation loss/accuracy curves
plt.figure(figsize=(12, 6))

# Training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
# %%
# Training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()