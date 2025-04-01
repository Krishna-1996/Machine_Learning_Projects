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

# Numerical features for the form (exams and marks)
entrance_exam_features = {
    'Mathexam': 'Mathematics',
    'Scienceexam_': 'Science',
    'Englishexam_': 'English',
}

marks_obtain_in_year_1 = {
    'Math191_': 'Mathematics Term I',
    'Science191_': 'Science Term I',
    'English191_': 'English Term I',
    'Math192_': 'Mathematics Term II',
    'Science192_': 'Science Term II',
    'English192_': 'English Term II',
    'Math193_': 'Mathematics Term III',
    'Science193_': 'Science Term III',
    'English193_': 'English Term III',
}

marks_obtain_in_year_2 = {
    'Math201_': 'Mathematics Term I',
    'Science201_': 'Science Term I',
    'English201_': 'English Term I',
    'Math202_': 'Mathematics Term II',
    'Science202_': 'Science Term II',
    'English202_': 'English Term II',
    'Math203_': 'Mathematics Term III',
    'Science203_': 'Science Term III',
    'English203_': 'English Term III',
}

# Home route to render the input form
@app.route('/')
def home():
    # Categorical features with better labels (questions)
    categorical_feature_labels = {
        'Gender': 'Select your Gender',
        'Age_as_of_Academic_Year_1718': 'Select your age',
        'Current_Year_1718': 'Select the grade you had in previous school',
        'Proposed_YearGrade_1819': 'Which grade you proposed for next year?',
        'Previous_Curriculum_17182': 'Select the Curriculum from previous school',
        'Current_School': 'Select your current school name',
        'Current_Curriculum': 'Select your current curriculum',
        'Previous_yearGrade': 'Select the curriculum system followed in previous school',
    }
    
    # Prepare a dictionary for passing dropdown options to the template
    features = {}
    for feature in encoding_dict:
        features[feature] = list(encoding_dict[feature].keys())  # Get unique values for dropdown options

    return render_template('index.html', 
                           features=features,                           
                           categorical_feature_labels=categorical_feature_labels,
                           entrance_exam_features=entrance_exam_features,
                           marks_obtain_in_year_1=marks_obtain_in_year_1,
                           marks_obtain_in_year_2=marks_obtain_in_year_2)


# Route to handle the prediction based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Fetch user inputs from the form
    user_input = {}
    for feature in encoding_dict:
        user_input[feature] = request.form.get(feature)

    # Fetch numerical feature values from the form
    numerical_input = []
    for feature in list(entrance_exam_features.keys()) + list(marks_obtain_in_year_1.keys()) + list(marks_obtain_in_year_2.keys()):
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
