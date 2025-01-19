import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from extract_and_save_features import extract_frames  # Assuming you have this function in extract_and_save_features.py
import cv2

app = Flask(__name__)

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1/Neural_Network_Model using Keras_Sequential.h5')

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route to serve index.html
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle video upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        # Secure the filename and save it
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the video and extract frames
        try:
            frames = extract_frames(filepath, frame_count=30)
        except Exception as e:
            return f'Error extracting frames: {str(e)}'
        
        if frames.shape[0] > 0:
            # Preprocess frames (e.g., resize, normalization) before prediction
            features = model.predict(frames)
            prediction = np.argmax(features, axis=1)  # Predict fake (1) or real (0)
            authenticity = np.max(features) * 100  # Percentage of authenticity (confidence score)
            
            result = "Fake" if prediction == 1 else "Real"
            
            # Render the result in the result.html template
            return render_template('result.html', result=result, authenticity=authenticity)
        else:
            return 'Error: No frames extracted from video'
    
    return 'Invalid file format'

if __name__ == '__main__':
    # Make sure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
