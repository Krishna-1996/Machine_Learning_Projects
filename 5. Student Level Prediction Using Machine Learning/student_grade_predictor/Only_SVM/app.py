import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained SVM model, scaler, and LabelEncoder mappings
svm_model_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\The_SVM_Model_Output.pkl'
scaler_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\The_SVM_Scaler.pkl'
label_encoders_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\Label_Encoders.pkl'

# Load the model, scaler, and label encoders
svm_model = joblib.load(svm_model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(label_encoders_path)

# Home route to render the input form
@app.route('/')
def home():
    # Define feature columns based on the encoded features
    features = {
        'Gender': ['Male', 'Female'],  # Assuming Gender as categorical
        'Grade': ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10'],
        'Curriculum': ['American', 'British'],  # Example for curriculum
        'Year_of_Admission': ['Current Student']  # Example for year of admission
    }

    # Add numerical features to the form as input fields
    numerical_features = ['Mathexam', 'Scienceexam_', 'Englishexam_', 'Math191_', 'Science191_', 'English191_',
                          'Math192_', 'Science192_', 'English192_', 'Math193_', 'Science193_', 'English193_',
                          'Math201_', 'Science201_', 'English201_', 'Math202_', 'Science202_', 'English202_',
                          'Math203_', 'Science203_', 'English203_']

    return render_template('index.html', features=features, numerical_features=numerical_features)

# Route to handle the prediction based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Fetch user inputs from the form
    user_input = {}
    for feature in ['Gender', 'Grade', 'Curriculum', 'Year_of_Admission']:
        user_input[feature] = request.form.get(feature)

    # Fetch numerical feature values from the form
    numerical_input = []
    for feature in ['Mathexam', 'Scienceexam_', 'Englishexam_', 'Math191_', 'Science191_', 'English191_',
                    'Math192_', 'Science192_', 'English192_', 'Math193_', 'Science193_', 'English193_',
                    'Math201_', 'Science201_', 'English201_', 'Math202_', 'Science202_', 'English202_',
                    'Math203_', 'Science203_', 'English203_']:
        numerical_input.append(float(request.form.get(feature, 0)))

    # Map the categorical user inputs to their corresponding numerical values using label encoders
    encoded_input = []
    for feature, value in user_input.items():
        if value in label_encoders[feature].classes_:
            encoded_input.append(label_encoders[feature].transform([value])[0])
        else:
            return "Error: Invalid input value."

    # Combine the encoded categorical input and numerical input
    final_input = np.array(encoded_input + numerical_input).reshape(1, -1)

    # Scale the features
    final_input_scaled = scaler.transform(final_input)

    # Make prediction using the SVM model
    prediction = svm_model.predict(final_input_scaled)[0]

    # Display message based on the prediction result
    if prediction == 0:
        result_message = "That's superb..!! The chosen grade is fit for the student, and with a little more effort, the student can achieve really good results."
    else:
        result_message = "No, the chosen grade for the student has some issues, but with a little practice and training, the student will do well."

    return render_template('result.html', result_message=result_message)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
