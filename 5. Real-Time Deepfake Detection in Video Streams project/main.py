# %%
import cv2
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
# Base directory path where the video folders are stored
base_dir = "D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1"  # Adjust this to your base directory

# %%
# Load pre-trained VGG16 model (without the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# %%
# Function to extract deep features using VGG16
def extract_deep_features(video_path):
    # Construct full path to the video
    full_video_path = os.path.join(base_dir, video_path)
    print(f"Attempting to open video: {full_video_path}")  # Debugging: print the full video path
    
    # Check if the video file exists
    if not os.path.exists(full_video_path):
        print(f"Error: Video file not found at {full_video_path}")
        return np.array([])  # Return empty array if file is not found

    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {full_video_path}")  # Debugging: print error if video can't be opened
        return np.array([])  # Return empty array if video can't be opened

    frames = []
    # Extract a few frames from the video
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            # Debugging: check the frame content before resizing
            print(f"Original frame shape: {frame.shape}, first 5 pixel values: {frame.flatten()[:5]}")  # Debugging print
            
            frame_resized = cv2.resize(frame, (224, 224))  # Resize to 224x224 for VGG16
            print(f"Resized frame shape: {frame_resized.shape}, first 5 pixel values: {frame_resized.flatten()[:5]}")  # Debugging print
            
            # Preprocess the frame for VGG16
            img_array = image.img_to_array(frame_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = base_model.predict(img_array)
            features = features.flatten()  # Flatten the features
            print(f"Extracted features shape: {features.shape}, first 10 values: {features[:10]}")  # Debugging print
            
            frames.append(features)
        else:
            print(f"Error: Unable to read frame from video {full_video_path}")
            break

    cap.release()
    
    # Check if any frames were extracted
    if len(frames) == 0:
        print(f"Warning: No frames read for video {full_video_path}")
        return np.array([])  # Return empty array if no frames were extracted
    
    # Return the feature array as a numpy array
    return np.array(frames).flatten()


# %%
# Collect features and labels
video_paths = ['YouTube-real/00170.mp4', 'Celeb-real/id1_0007.mp4', 'Celeb-synthesis/id1_id0_0007.mp4']  # Example paths
features = []
labels = []

# %%
for video_path in video_paths:
    feature_vector = extract_deep_features(video_path)
    
    if feature_vector.size > 0:
        features.append(feature_vector)
        
        # Label assignment based on folder
        folder_name = video_path.split('/')[0]
        if folder_name == "Celeb-real":
            labels.append(0)  # 0 for real videos
        elif folder_name == "Celeb-synthesis":
            labels.append(1)  # 1 for synthetic videos
        else:  # YouTube-real
            labels.append(0)  # 0 for real videos

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# %%
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Build a Deep Learning Model (Simple Feedforward Neural Network)
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# %%
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# %%
print(f"Feature vector shape for video {video_path}: {feature_vector.shape}")
print(f"Feature vector for {video_path}: {feature_vector[:10]}")  # Print first 10 values

# %%
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy score: {accuracy}")

# %%
