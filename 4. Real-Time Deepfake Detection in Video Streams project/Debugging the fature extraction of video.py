import cv2
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Base directory path where the video folders are stored
base_dir = "D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1"  # Change this to your base directory

# Load pre-trained VGG16 model (without the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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
    
    # Check if any frames were extracted
    if len(frames) == 0:
        print(f"Warning: No frames read for video {full_video_path}")
        return np.array([])  # Return empty array if no frames were extracted
    
    # Return the feature array as a numpy array
    return np.array(frames).flatten()

# Example usage: Try to extract deep features from a video path in the "YouTube-real" folder
video_path = 'YouTube-real/00170.mp4'  # This path comes from your CSV or other source
features = extract_deep_features(video_path)

# Check if features were extracted
if features.size > 0:
    print(f"Extracted features for {video_path} with shape: {features.shape}")
else:
    print(f"No features extracted for {video_path}")
