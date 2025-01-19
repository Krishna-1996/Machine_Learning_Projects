import os
import numpy as np
import pickle
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import pandas as pd

# Path to main directory and CSV file
base_dir = r"D:\MSc. Project DeepFake Detection Datasets\Celeb-DF-v1"
csv_file = os.path.join(base_dir, "Video_Label_and_Dataset_List.csv")

processed_data_dir = os.path.join(base_dir, "Updated_processed_data")
os.makedirs(processed_data_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)

# Pretrained model - VGG16 for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_frames(video_path, frame_count=30, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Error: No frames in video {video_path}")
        return np.array([])

    interval = max(total_frames // frame_count, 1)
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, target_size)
            frame = preprocess_input(frame)
            frames.append(frame)
    cap.release()
    return np.array(frames)

y_data = []
for idx, row in df.iterrows():
    video_path = row['Video Path']
    label = 1 if row['Label'] == 'fake' else 0
    full_video_path = os.path.join(base_dir, video_path)

    print(f"Processing video {idx + 1}/{len(df)}: {video_path}")
    frames = extract_frames(full_video_path)

    if len(frames) == 0:
        print(f"Skipping video {video_path} due to frame extraction failure")
        continue

    features = base_model.predict(frames)
    features = features.reshape(features.shape[0], -1).mean(axis=0)

    feature_file = os.path.join(processed_data_dir, f'features_{idx}.npy')
    np.save(feature_file, features)

    y_data.append(label)

labels_file = os.path.join(processed_data_dir, 'labels.pkl')
with open(labels_file, 'wb') as f:
    pickle.dump(y_data, f)

print("Feature extraction and saving completed!")