# Database: D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1

# %%

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import random
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
# Base directory where the videos are located
base_dir = r"D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1"  # Replace with your actual base directory
video_path = 'YouTube-real/00170.mp4'  # This is from CSV or input data
# %%
# Load the test video list from the CSV
csv_file = os.path.join(base_dir, "Test_Videos_List.csv")  # Full path to the CSV
df = pd.read_csv(csv_file, header=None)
video_paths = df[0].tolist()


# %%
# Define a function to extract features from videos
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return []  # Return empty list if video can't be opened
    
    # Read some frames from the video (e.g., first 5 frames)
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (64, 64))
            frames.append(np.mean(frame_resized))  # Using mean pixel value as feature
            
    cap.release()
    
    if len(frames) == 0:
        print(f"Warning: No frames read for video {video_path}")
        return []  # Return empty if no frames were read
    
    return np.array(frames).flatten()

# Create a list of features and labels
features = []
labels = []

# %%
# Loop through all video paths and extract features and assign labels based on folder
for video_path in video_paths:
    # Extract the folder name (e.g., Celeb-real, Celeb-synthesis, or YouTube-real)
    folder_name = video_path.split('/')[0]
    
    # Construct the full path to the video file
    full_video_path = os.path.join(base_dir, video_path)  # Full path
    
    # Extract features for the current video
    feature_vector = extract_features(full_video_path)
    
    if len(feature_vector) > 0:
        features.append(feature_vector)
        
        # Label assignment based on folder
        if folder_name == "Celeb-real":
            labels.append(0)  # 0 for real videos
        elif folder_name == "Celeb-synthesis":
            labels.append(1)  # 1 for synthetic videos
        else:  # YouTube-real
            labels.append(0)  # 0 for real videos
    else:
        print(f"Skipping video {video_path} due to extraction failure")

# Convert the features and labels to numpy arrays
X = np.array(features)
y = np.array(labels)

# %%
# Ensure that X is not empty and has at least 1 feature
if X.shape[1] == 0:
    print("Error: No valid features extracted. Exiting.")
else:
    # Perform a train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the Random Forest model: {accuracy:.4f}")

# %%

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load pre-trained VGG16 model (without the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to extract deep features using VGG16
def extract_deep_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return np.array([])  # Return empty array if video can't be opened
    
    # Extract a few frames from the video
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (224, 224))  # Resize to 224x224 for VGG16
            # Preprocess the frame for VGG16
            img_array = image.img_to_array(frame_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = base_model.predict(img_array)
            features = features.flatten()  # Flatten the features
            frames.append(features)
        else:
            break
    
    cap.release()
    
    if len(frames) == 0:
        print(f"Warning: No frames read for video {video_path}")
        return np.array([])  # Return empty array if no frames were read
    
    # Return the feature array as a numpy array
    return np.array(frames).flatten()

# Example: Extract deep features from a video
video_path = 'YouTube-real/00170.mp4'
features = extract_deep_features(video_path)

if features.size > 0:  # Check if features were extracted
    print(features.shape)  # Check the shape of extracted features
else:
    print("No features extracted")

# %%



# %%


# %%




# %%


# %%

