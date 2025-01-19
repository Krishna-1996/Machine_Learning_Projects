import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50

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

# ####################### Transfer Learning with ResNet50 #######################
# Load ResNet50 with pre-trained ImageNet weights, excluding the top classification layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

input_shape = X_train.shape[1:] 
# Define the model using Dense layers
input_layer = Input(shape=input_shape)
x = Dense(512, activation='relu')(input_layer)
x = Dropout(0.5)(x)  # Dropout to reduce overfitting
x = Dense(2, activation='softmax')(x)  # Softmax for binary classification (real or fake)

# Define the model
model = Model(inputs=input_layer, outputs=x)

# Freeze the base model layers (optional for fine-tuning)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',  # Categorical loss for multi-class classification
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# ########################## Save the Model ##########################
model_save_path = os.path.join(base_dir, 'Model_2 ResNet50_Transfer_Learning.h5')
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
