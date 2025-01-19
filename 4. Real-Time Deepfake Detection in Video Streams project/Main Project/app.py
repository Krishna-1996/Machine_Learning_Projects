import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained model and labels
model_path = 'path_to_your_model.h5'  # Update this path
model = load_model(model_path)

# Load label encoder (if applicable)
labels_file = 'path_to_your_labels.pkl'  # Update this path if you are using label encoding
with open(labels_file, 'rb') as f:
    label_encoder = pickle.load(f)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Path for saving uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract frames from the video
def extract_frames(video_path, frame_count=30, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
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

# Route for uploading and processing the video
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Extract frames and predict using the model
        frames = extract_frames(video_path)
        if len(frames) == 0:
            return jsonify({'error': 'Failed to extract frames from the video'}), 400
        
        features = model.predict(frames)
        features = features.reshape(features.shape[0], -1).mean(axis=0)

        # Predict class (fake or real)
        prediction = model.predict(np.expand_dims(features, axis=0))
        prediction_class = np.argmax(prediction, axis=1)[0]
        authenticity = prediction[0][prediction_class] * 100  # Percentage of authenticity

        # Convert to label
        label = 'Fake' if prediction_class == 1 else 'Real'
        return jsonify({
            'prediction': label,
            'authenticity_level': f'{authenticity:.2f}%',
        })

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
