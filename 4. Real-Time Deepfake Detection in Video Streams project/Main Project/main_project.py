import os
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Paths
base_dir = r"D:\MSc. Project DeepFake Detection Datasets\Celeb-DF-v1"
processed_data_dir = os.path.join(base_dir, "processed_data")

# Load pre-extracted features and labels
X_data = []
y_data = []

df = pd.read_csv(os.path.join(base_dir, "Video_Label_and_Dataset_List.csv"))
for idx, row in df.iterrows():
    feature_file = os.path.join(processed_data_dir, f'features_{idx}.npy')
    if os.path.exists(feature_file):
        features = np.load(feature_file)
        X_data.append(features)
        y_data.append(row['Label'])

X_data = np.array(X_data)
y_data = np.array([1 if label == 'fake' else 0 for label in y_data])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(512, activation='relu', kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {metrics[1] * 100:.2f}%")

# Metrics
predictions = np.argmax(model.predict(X_test), axis=1)
true_labels = np.argmax(y_test, axis=1)
print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=['Real', 'Fake']))

cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()