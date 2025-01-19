# %%
# Import necessary libraries
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input


# %%
# Path to the main directory
base_dir = r"D:\MSc. Project DeepFake Detection Datasets\Celeb-DF-v1"
train_sample_dir = os.path.join(base_dir, "train_sample")
test_sample_dir = os.path.join(base_dir, "test_sample")

# %%
# Load the CSV file with video paths and labels
csv_file = os.path.join(base_dir, "Video_Label_and_Dataset_List.csv")
df = pd.read_csv(csv_file)

# %%
# Check class distribution
class_counts = df['Label'].value_counts()
print(f"Class distribution: \n{class_counts}")
class_weights = {0: len(df) / (2 * class_counts[0]), 1: len(df) / (2 * class_counts[1])}
print(f"Class weights: {class_weights}")

# %%
# Prepare the data
X_data = []
y_data = []

# %%
# Function to load and preprocess video frames (Replace with actual frame extraction if not done yet)
import cv2
def extract_frames(video_path, frame_count=30, target_size=(224, 224)):
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return None
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // frame_count  # Extract `frame_count` evenly spaced frames

    # Read the frames
    for i in range(frame_count):
        frame_id = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            # Resize and normalize the frame
            frame = cv2.resize(frame, target_size)
            frame = preprocess_input(frame)  # VGG16 preprocessing
            frames.append(frame)
    
    cap.release()

    if len(frames) > 0:
        print(f"Extracted {len(frames)} frames from {video_path}.")
    else:
        print(f"No frames extracted from {video_path}.")
    
    return np.array(frames)

# %%
# Iterate through the CSV and load corresponding video data
for idx, row in df.iterrows():
    video_path = row['Video Path']
    label = 1 if row['Label'] == 'fake' else 0  # Fake -> 1, Real -> 0
    
    # Create full video path for train/test samples
    full_video_path = os.path.join(base_dir, video_path)
    
    # Extract frames (Make sure this step is correct and efficient)
    frames = extract_frames(full_video_path, frame_count=30)
    
    # If frame extraction returns empty, skip
    if frames is None or len(frames) == 0:
        continue
    
    X_data.append(frames)
    y_data.append(label)

# %%
# Convert X_data and y_data into NumPy arrays
X_data = np.array(X_data)
y_data = np.array(y_data)

# %%
# Check if the data shapes are correct
print(f"Shape of X_data: {X_data.shape}")
print(f"Shape of y_data: {y_data.shape}")

# %%
# Ensure frames data is consistent
# Normalize the input features (pixel values range from 0-255)
X_data = X_data / 255.0

# %%
# Split data into train and test sets
if X_data.shape[0] == 0:
    print("Error: No data available after frame extraction. Please check your extract_frames function.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# %%
# One-hot encode the labels
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# %%
# #############################################

from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

# Build the model
model = Sequential()

# Input layer specifying the shape (30 frames, 224x224 RGB)
model.add(Input(shape=(30, 224, 224, 3)))  # Shape of the input data (30 frames, 224x224 RGB)

# Apply TimeDistributed Conv2D and MaxPooling2D
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')))  # Conv2D with 3x3 filter, 32 filters
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))  # MaxPooling2D with 2x2 pool size

# Flatten each frame separately
model.add(TimeDistributed(Flatten()))  # Flatten each frame individually

# Print the model summary to see the shape of the output after these layers
model.summary()

# Proceed with the rest of the layers
model.add(LSTM(256, return_sequences=False))  # LSTM expects a 3D input: (batch_size, timesteps, features)
model.add(Dropout(0.3))

model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

# Output layer with softmax activation
model.add(Dense(2, activation='softmax'))  # 2 output classes: real and fake

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

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

# %%
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# %%
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
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# %%
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
