# model.py (Model logic for loading the model and making predictions)

from tensorflow.keras.models import load_model
import numpy as np

# Load your trained model (make sure this path points to your model)
model = load_model(r'D:/Machine_Learning_Projects/5. Real-Time Deepfake Detection in Video Streams project/my_webpage/models/model.h5')

def predict_video_features(features):
    """
    Predict if a video is real or fake using the extracted features and the trained model.
    """
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    prediction = model.predict(features)
    return prediction  # Return prediction (you can return both the class and the confidence)
