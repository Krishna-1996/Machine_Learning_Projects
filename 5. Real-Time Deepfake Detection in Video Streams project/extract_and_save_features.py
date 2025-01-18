import os
import numpy as np
import pickle
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Path to main directory and CSV file
base_dir = r"D:\MSc. Project DeepFake Detection Datasets\Celeb-DF-v1"
csv_file = os.path.join(base_dir, "Video_Label_and_Dataset_List.csv")

# Directory to save processed features
processed_data_dir = os.path.join(base_dir, "processed_data")
os.makedirs(processed_data_dir, exist_ok=True)

# Load CSV
import pandas as pd
df = pd.read_csv(csv_file)

# Pretrained model - VGG16 for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to extract frames
def extract_frames(video_path, frame_count=30, target_size=(224, 224)):
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
    return np.array(frames)

# Iterate through the CSV and extract/save features
y_data = []  # Store labels
for idx, row in df.iterrows():
    video_path = row['Video Path']
    label = 1 if row['Label'] == 'fake' else 0  # Fake -> 1, Real -> 0
    
    # Create full video path
    full_video_path = os.path.join(base_dir, video_path)
    
    # Extract frames
    print(f"Processing video {idx + 1}/{len(df)}: {video_path}")
    frames = extract_frames(full_video_path, frame_count=30)
    
    # Extract features using VGG16
    features = base_model.predict(frames)
    features = features.reshape(features.shape[0], -1).flatten()  # Flatten all features
    
    # Save features to a file
    feature_file = os.path.join(processed_data_dir, f'features_{idx}.npy')
    np.save(feature_file, features)
    
    # Append label
    y_data.append(label)

# Save labels
labels_file = os.path.join(processed_data_dir, 'labels.pkl')
with open(labels_file, 'wb') as f:
    pickle.dump(y_data, f)

print("Feature extraction and saving completed!")
