import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from extract_and_save_features import extract_frames  # Import your frame extraction function
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model
model_path = 'D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1/Neural_Network_Model using Keras_Sequential.h5'
model = load_model(model_path)

# Load VGG16 model for feature extraction
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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
        frames = extract_frames(filepath, frame_count=30)

        if frames.shape[0] > 0:
            # Use VGG16 to extract features (flatten and preprocess the frames)
            features = vgg16_model.predict(frames)  # Output will be of shape (30, 7, 7, 512)
            
            # Flatten the features
            features = features.reshape(features.shape[0], -1)  # Now shape is (30, 25088)
            
            # If needed, you can average the features across frames to get a single feature vector
            features = features.mean(axis=0)  # Shape will be (25088,)

            # Use the trained model to predict the label
            prediction = model.predict(np.expand_dims(features, axis=0))  # Add batch dimension
            prediction_class = np.argmax(prediction, axis=1)  # 0 -> Real, 1 -> Fake
            
            authenticity = np.max(prediction) * 100  # Confidence score (percentage)

            # Show result
            result = "Fake" if prediction_class == 1 else "Real"
            return render_template('result.html', result=result, authenticity=authenticity)
        else:
            return 'Error: No frames extracted from video'
    
    return 'Invalid file format'

if __name__ == '__main__':
    # Make sure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
