import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers

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

# Custom Positional Embedding Layer
class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        # Create learnable positional embeddings
        self.positional_embeddings = self.add_weight(
            name="positional_embeddings", 
            shape=(self.num_patches, self.embedding_dim), 
            initializer="random_normal"
        )

    def call(self, inputs):
        # Add positional embeddings to input patches
        return inputs + self.positional_embeddings

# Vision Transformer Model
def build_vit_model(input_shape):
    inputs = Input(shape=input_shape)
    patch_size = 16
    num_patches = input_shape[0] // patch_size
    
    # Patch embedding (Linear projection)
    patch_projection = Dense(128, activation='relu')(inputs)

    # Add positional embeddings using the custom layer
    position_embedding_layer = PositionalEmbedding(num_patches, 128)
    x = position_embedding_layer(patch_projection)

    # Transformer blocks
    for _ in range(4):  # 4 transformer layers
        x1 = LayerNormalization()(x)
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x1, x1)
        x2 = Dropout(0.1)(attention_output + x)
        x3 = LayerNormalization()(x2)
        x = Dense(128, activation='relu')(x3)

    x = tf.reduce_mean(x, axis=1)  # Global average pooling
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Build and compile the model
vit_model = build_vit_model((X_train.shape[1],))  # Shape of the input (number of features)
vit_model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Train the ViT model
history_vit = vit_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# Save the trained model
vit_model_path = os.path.join(base_dir, 'Model_3_Using_Vision_Transformer_ViT.h5')
vit_model.save(vit_model_path)
print(f"Vision Transformer model saved at: {vit_model_path}")

# Evaluate the model on the test set
metrics = vit_model.evaluate(X_test, y_test)
print(f"Test Accuracy (ViT): {metrics[1] * 100:.2f}%")

# Additional: If you want to print classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

# Predict on test set
y_pred = vit_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# Optional: Plot training history (accuracy and loss)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history_vit.history['accuracy'])
plt.plot(history_vit.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_vit.history['loss'])
plt.plot(history_vit.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

