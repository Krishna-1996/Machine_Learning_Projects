import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained SVM model and LabelEncoder mappings
svm_model_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\The_SVM_Model_Output.pkl'
model = joblib.load(svm_model_path)

# Load the feature encoding information from the Excel file
encoding_excel_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\feature_encoding_info.xlsx'
encoding_df = pd.read_excel(encoding_excel_path)

# Create a dictionary for fast lookup of feature encodings
encoding_dict = {}
for _, row in encoding_df.iterrows():
    feature = row['Feature Name']
    if feature not in encoding_dict:
        encoding_dict[feature] = {}
    encoding_dict[feature][row['Unique Value']] = row['Numerical Value']

# Home route to render the input form
@app.route('/')
def home():
    # Fetch feature names and unique values to display as options in the form
    features = {}
    for feature in encoding_dict:
        features[feature] = list(encoding_dict[feature].keys())
    return render_template('index.html', features=features)

# Route to handle the prediction based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Fetch user inputs from the form
    user_input = {}
    for feature in encoding_dict:
        user_input[feature] = request.form.get(feature)

    # Map the categorical user inputs to their corresponding numerical values
    numerical_input = []
    for feature, value in user_input.items():
        if value in encoding_dict[feature]:
            numerical_input.append(encoding_dict[feature][value])
        else:
            return "Error: Invalid input value."

    # Create the input array for prediction (make sure to match the model input)
    # Assumes that all features are provided by the user, matching the model's expected feature set
    numerical_input = np.array(numerical_input).reshape(1, -1)

    # Make prediction using the SVM model
    prediction = model.predict(numerical_input)[0]

    # Display message based on the prediction result
    if prediction == 0:
        result_message = "That's superb..!! The chosen grade is fit for the student, and with little more effort, the student can achieve really good results."
    else:
        result_message = "No, the chosen grade for the student has some issues, but with a little practice and training, the student will do well."

    return render_template('result.html', result_message=result_message)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
