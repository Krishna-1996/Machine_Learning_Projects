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
# Load the test video list from the CSV
csv_file = "D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1/Test_Videos_List.csv"
df = pd.read_csv(csv_file, header=None)
video_paths = df[0].tolist()

# %%
# Define a function to extract features from videos
# For the purpose of this example, we'll just use some basic feature extraction (like mean pixel values).
def extract_features(video_path):
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Read some frames from the video (e.g., first 5 frames)
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            # Resize the frame and extract color histogram or mean pixel values
            frame_resized = cv2.resize(frame, (64, 64))
            frames.append(np.mean(frame_resized))  # Using mean pixel value as feature
            
    cap.release()
    
    # Flatten the list of features (in this case, 5 mean values from 5 frames)
    return np.array(frames).flatten()

# Create a list of features and labels
features = []
labels = []

# %%
# Loop through all video paths and extract features and assign labels based on folder
for video_path in video_paths:
    # Extract the label (folder name)
    folder_name = video_path.split('/')[0]
    
    # Extract features for the current video
    feature_vector = extract_features(video_path)
    features.append(feature_vector)
    
    # Label assignment based on folder
    if folder_name == "Celeb-real":
        labels.append(0)  # 0 for real videos
    elif folder_name == "Celeb-synthesis":
        labels.append(1)  # 1 for synthetic videos
    else:  # YouTube-real
        labels.append(0)  # 0 for real videos

# Convert the features and labels to numpy arrays
X = np.array(features)
y = np.array(labels)

# %%
# Perform a train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest model: {accuracy:.4f}")

# %%



# %%


# %%




# %%


# %%

