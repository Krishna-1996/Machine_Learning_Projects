# app.py (Backend)

from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from model import predict_video_features

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
BASE_DATA_DIR = r"D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1"  # Data directory

def extract_video_features(video_path):
    """
    Extract features from the video using pre-trained VGG16 model.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(5):  # Read 5 frames from the video
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (224, 224))  # Resize for VGG16
            img_array = image.img_to_array(frame_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Feature extraction using VGG16 model
            features = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).predict(img_array)
            features = features.flatten()
            frames.append(features)
        else:
            break
    cap.release()
    return np.array(frames).flatten()

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file found'}), 400

    video_file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    features = extract_video_features(video_path)
    prediction = predict_video_features(features)  # Call the model for prediction

    return jsonify({
        'prediction': prediction[0],  # Assuming model returns the class label
        'confidence': prediction[1]   # Confidence score
    })

if __name__ == '__main__':
    app.run(debug=True)
