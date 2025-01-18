import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

# Directory to load processed features
base_dir = r"D:\MSc. Project DeepFake Detection Datasets\Celeb-DF-v1"
processed_data_dir = os.path.join(base_dir, "processed_data")

# Load features and labels
X_data = []
y_data = []

# Iterate through saved feature files
for file in sorted(os.listdir(processed_data_dir)):
    if file.startswith('features_') and file.endswith('.npy'):
        feature_file = os.path.join(processed_data_dir, file)
        features = np.load(feature_file)
        X_data.append(features)

# Load labels
labels_file = os.path.join(processed_data_dir, 'labels.pkl')
with open(labels_file, 'rb') as f:
    y_data = pickle.load(f)

# Convert to NumPy arrays
X_data = np.array(X_data)
y_data = np.array(y_data)

print(f"Loaded features shape: {X_data.shape}")
print(f"Loaded labels shape: {y_data.shape}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Build a simple deep learning model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # L2 Regularization
model.add(Dense(2, activation='softmax'))  # 2 output classes: real and fake

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
