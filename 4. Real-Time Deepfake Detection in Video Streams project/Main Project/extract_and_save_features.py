import os
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input

# Function to extract frames from a video
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
