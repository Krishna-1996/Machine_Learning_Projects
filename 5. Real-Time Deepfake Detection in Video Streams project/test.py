# %% 
# Import necessary libraries and modules.
import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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
# Pretrained model - VGG16 for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# %% 
# Function to load and preprocess video frames
def extract_frames(video_path, frame_count=30, target_size=(224, 224)):
    # Load video using OpenCV
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
            frame = frame.astype('float32') / 255.0
            frames.append(frame)
    cap.release()
    return np.array(frames)

# %%
# Prepare data for training and testing
X_data = []
y_data = []

# %% 
# Iterate through the CSV and load corresponding video data
for idx, row in df.iterrows():
    video_path = row['Video Path']
    label = 1 if row['Label'] == 'fake' else 0  # Fake -> 1, Real -> 0
    
    # Create full video path for train/test samples
    if video_path.startswith("train_sample"):
        full_video_path = os.path.join(base_dir, video_path)
    else:
        full_video_path = os.path.join(base_dir, video_path)
    
    # Extract frames and features using the VGG16 model
    frames = extract_frames(full_video_path, frame_count=30)  # Ensure consistent frame count
    
    # Convert frames to arrays for neural network input
    frames = np.array([img_to_array(frame) for frame in frames])
    
    # Extract features using VGG16 (without the top classification layer)
    features = base_model.predict(frames)
    
    # Flatten features to a 1D vector per frame
    features = features.reshape(features.shape[0], -1)  # Flatten each frame to 1D
    
    # Check the shape of the features before appending
    print(f"Features shape for video {video_path}: {features.shape}")
    
    # Ensure that features are consistent in shape
    X_data.append(features)
    y_data.append(label)

# Convert X_data and y_data into NumPy arrays
X_data = np.array(X_data, dtype=object)  # Object type to allow for variable-sized arrays
y_data = np.array(y_data)

# Ensure X_data has a consistent shape
print(f"Shape of X_data: {X_data.shape}")

# %% 
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# %% 
# One-hot encode the labels
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# %% 
# Build a simple deep learning model for classification
model = Sequential()
model.add(Flatten(input_shape=(X_train.shape[1],)))  # Flatten input
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 2 output classes: real and fake

# %% 
# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



# %% 
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# %% 
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# You can also check model performance on the test set and plot loss/accuracy if needed

# %%
