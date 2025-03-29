import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
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
    encoding_dict[feature][str(row['Unique Value'])] = row['Numerical Value']  # Ensure that keys are strings

# Home route to render the input form
@app.route('/')
def home():
    # Categorical features with better labels (questions)
    feature_labels = {
        'Gender': 'What is your gender?',
        'Age_as_of_Academic_Year_1718': 'What is your age as of academic year 2017-2018?',
        'Current_Year_1718': 'What grade are you in for the current year (2017-2018)?',
        'Parental_education': 'What is the highest level of education completed by your parents?',
        'Region': 'Which region do you live in?',
        # Add more features as required with meaningful names
    }
    
    features = {}
    for feature in encoding_dict:
        features[feature] = list(encoding_dict[feature].keys())

    # Numerical features: entrance exam scores, and marks in previous years (2019/2020)
    entrance_exam_features = ['Mathexam', 'Scienceexam_', 'Englishexam_']
    marks_previous_years_1 = ['Math191_', 'Science191_', 'English191_']
    marks_previous_years_2 = ['Math192_', 'Science192_', 'English192_']
    marks_previous_years_3 = ['Math193_', 'Science193_', 'English193_']
    marks_previous_years_4 = ['Math201_', 'Science201_', 'English201_']
    marks_previous_years_5 = ['Math202_', 'Science202_', 'English202_']
    marks_previous_years_6 = ['Math203_', 'Science203_', 'English203_']

    return render_template('index.html', features=features, feature_labels=feature_labels, 
                           entrance_exam_features=entrance_exam_features, 
                           marks_previous_years_1=marks_previous_years_1, marks_previous_years_2=marks_previous_years_2, 
                           marks_previous_years_3=marks_previous_years_3, marks_previous_years_4=marks_previous_years_4, 
                           marks_previous_years_5=marks_previous_years_5, marks_previous_years_6=marks_previous_years_6)

# Route to handle the prediction based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Fetch user inputs from the form
    user_input = {}
    for feature in encoding_dict:
        user_input[feature] = request.form.get(feature)

    # Fetch numerical feature values from the form
    numerical_input = []
    for feature in ['Mathexam', 'Scienceexam_', 'Englishexam_',
                    'Math191_', 'Science191_', 'English191_',
                    'Math192_', 'Science192_', 'English192_',
                    'Math193_', 'Science193_', 'English193_',
                    'Math201_', 'Science201_', 'English201_',
                    'Math202_', 'Science202_', 'English202_',
                    'Math203_', 'Science203_', 'English203_']:
        numerical_input.append(float(request.form.get(feature, 0)))

    # Map the categorical user inputs to their corresponding numerical values
    encoded_input = []
    for feature, value in user_input.items():
        if value in encoding_dict[feature]:
            encoded_input.append(encoding_dict[feature][value])
        else:
            print(f"Invalid input value for {feature}: {value}. Check the available options.")
            return f"Error: Invalid input value for {feature}. Check the available options."

    # Combine the encoded categorical input and numerical input
    final_input = np.array(encoded_input + numerical_input).reshape(1, -1)

    # Make prediction using the SVM model
    prediction = model.predict(final_input)[0]

    # Display message based on the prediction result
    if prediction == 0:
        result_message = "That's superb..!! The chosen grade is fit for the student, and with little more effort, the student can achieve really good results."
    else:
        result_message = "No, the chosen grade for the student has some issues, but with a little practice and training, the student will do well."

    return render_template('result.html', result_message=result_message)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
