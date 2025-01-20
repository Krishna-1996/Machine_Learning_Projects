import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
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

# ####################### Hybrid CNN + ViT Model #######################
# Adjust input shape for pre-extracted features (flattened)
input_shape = (25088,)  # Shape of pre-extracted features (flattened)

# Define the CNN model for feature extraction
cnn_input = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3), activation='relu')(cnn_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# Load Vision Transformer model from TensorFlow Hub (updated model URL)
vit_model_url = "https://tfhub.dev/google/vit_small_patch16_224/1"  # New ViT model URL
vit_layer = hub.KerasLayer(vit_model_url, trainable=False, name="vit_layer")

# CNN features concatenated with ViT
vit_input = Input(shape=(25088,))  # For hybrid approach
cnn_vit_combined = tf.concat([x, vit_input], axis=-1)  # Concatenate CNN features and ViT features

# Add dense layers after combining
x = Dense(512, activation='relu')(cnn_vit_combined)
x = Dropout(0.5)(x)  # Dropout to reduce overfitting
x = Dense(2, activation='softmax')(x)  # Softmax for binary classification (real or fake)

# Define the model
model = Model(inputs=[cnn_input, vit_input], outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# ########################## Model Training ##########################
X_train_cnn = np.array([np.reshape(x, (224, 224, 3)) for x in X_train])  # Assuming raw images for CNN
X_test_cnn = np.array([np.reshape(x, (224, 224, 3)) for x in X_test])  # Assuming raw images for CNN

# Train the model
history = model.fit([X_train_cnn, X_train], y_train, epochs=50, batch_size=32, validation_data=([X_test_cnn, X_test], y_test), callbacks=callbacks)

# ########################## Save the Model ##########################
model_save_path = os.path.join(base_dir, 'Model_Hybrid_CNN_ViT.h5')
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# ########################## Evaluate the Model ##########################
metrics = model.evaluate([X_test_cnn, X_test], y_test)
print(f"Test Accuracy: {metrics[1] * 100:.2f}%")

# ########################## Metrics and Reporting ##########################
predictions = np.argmax(model.predict([X_test_cnn, X_test]), axis=1)
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
