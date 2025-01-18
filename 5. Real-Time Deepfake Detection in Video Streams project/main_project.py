
# %%
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# %%
# Directory to load processed features
base_dir = r"D:\MSc. Project DeepFake Detection Datasets\Celeb-DF-v1"
processed_data_dir = os.path.join(base_dir, "processed_data")

# %%
# Load features and labels
print("Loading features and labels...")
X_data = []
for file in sorted(os.listdir(processed_data_dir)):
    if file.startswith('features_') and file.endswith('.npy'):
        feature_file = os.path.join(processed_data_dir, file)
        features = np.load(feature_file)
        X_data.append(features)

# Load labels
labels_file = os.path.join(processed_data_dir, 'labels.pkl')
with open(labels_file, 'rb') as f:
    y_data = pickle.load(f)

# %%
# Convert to NumPy arrays
X_data = np.array(X_data)
y_data = np.array(y_data)

# %%
print(f"Loaded features shape: {X_data.shape}")
print(f"Loaded labels shape: {y_data.shape}")

# %%
# Reshape X_data for temporal modeling (LSTM)
frame_count = 30  # Assuming each video has 30 frames
X_data = X_data.reshape(X_data.shape[0], frame_count, -1)  # Shape: (num_samples, 30, features_per_frame)

# %%
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# %%
# One-hot encode the labels
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# %%
# Compute class weights to address imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_data),
    y=y_data
)
class_weights = dict(enumerate(class_weights))

# %%
# Build the updated Bidirectional LSTM model
model = Sequential()
model.add(Input(shape=(frame_count, X_data.shape[2])))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))  # Increased dropout rate
model.add(Dense(2, activation='softmax'))  # 2 output classes: real and fake

# %%
# Compile the model with a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# %%
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Start with a slightly higher learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %%
# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# %%
# Train the model with enhanced regularization and learning rate scheduling
print("Training the model with enhancements...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stopping, lr_scheduler]
)
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# %%
# Save the model in the recommended Keras format
# model_save_path = os.path.join(base_dir, "deepfake_lstm_model.keras")
# model.save(model_save_path)
# print(f"Model saved at: {model_save_path}")

# %%
# Save training history for future analysis
# history_save_path = os.path.join(base_dir, "training_history.pkl")
# with open(history_save_path, 'wb') as f:
#     pickle.dump(history.history, f)
# print(f"Training history saved at: {history_save_path}")

# %%
# Plot the training history
print("Plotting training history...")
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Evaluate additional metrics like classification report and confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# %%
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

# %%
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
