import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Paths
base_dir = r"D:\MSc. Project DeepFake Detection Datasets\Celeb-DF-v1"
Updated_processed_data_dir = os.path.join(base_dir, "Updated_processed_data")

# Load pre-extracted features and labels
X_data = []
y_data = []

df = pd.read_csv(os.path.join(base_dir, "Video_Label_and_Dataset_List.csv"))
for idx, row in df.iterrows():
    feature_file = os.path.join(Updated_processed_data_dir, f'features_{idx}.npy')
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

# Ensure input shape compatibility
input_shape = (224, 224, 3)  # Vision Transformer requires 224x224x3 input images

# Adjust features to match ViT input shape
from tensorflow.image import resize

def resize_data(data, target_shape):
    # Ensure data has 4 dimensions (num_samples, height, width, channels)
    if len(data.shape) == 4:
        resized_data = np.array([resize(img, target_shape[:2]).numpy() for img in data])
    elif len(data.shape) == 3:  # Add channel dimension if missing
        resized_data = np.array([resize(np.expand_dims(img, axis=-1), target_shape[:2]).numpy() for img in data])
    else:
        raise ValueError("Input data must have 3 or 4 dimensions.")
    return resized_data

# Reshape and normalize input data
try:
    X_train = resize_data(X_train, input_shape)
    X_test = resize_data(X_test, input_shape)
except ValueError as e:
    print(f"Error during resizing: {e}")
    exit()

X_train = X_train / 255.0
X_test = X_test / 255.0

# ####################### Transfer Learning with Vision Transformer #######################
# Load Vision Transformer model from TensorFlow Hub
vit_model_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
vit_layer = hub.KerasLayer(vit_model_url, trainable=False, name="vit_layer")

# Define the model using Dense layers
input_layer = Input(shape=input_shape)
x = vit_layer(input_layer)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout to reduce overfitting
x = Dense(2, activation='softmax')(x)  # Softmax for binary classification (real or fake)

# Define the model
model = Model(inputs=input_layer, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# ########################## Save the Model ##########################
model_save_path = os.path.join(base_dir, 'Model_ViT_Transfer_Learning.h5')
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# ########################## Evaluate the Model ##########################
metrics = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {metrics[1] * 100:.2f}%")

# ########################## Metrics and Reporting ##########################
predictions = np.argmax(model.predict(X_test), axis=1)
true_labels = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=['Real', 'Fake']))

cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)

# ########################## Plot Training History ##########################
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
