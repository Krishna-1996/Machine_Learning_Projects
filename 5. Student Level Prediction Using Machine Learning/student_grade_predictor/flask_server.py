from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the trained SVM model
model_path = 'D:/Machine_Learning_Projects/5. Student Level Prediction Using Machine Learning/student_grade_predictor/svm_model.pkl'
svm_model = joblib.load(model_path)

# Load the label encoding mappings from the Excel sheet
label_mappings = pd.read_excel('D:/Machine_Learning_Projects/5. Student Level Prediction Using Machine Learning/student_grade_predictor/feature_encoding_info.xlsx')

# Create a dictionary for each feature to hold its mappings
mappings = {}

for feature in label_mappings['Feature Name'].unique():
    feature_map = label_mappings[label_mappings['Feature Name'] == feature]
    mappings[feature] = dict(zip(feature_map['Unique Value'], feature_map['Numerical Value']))

# Initialize Flask app
app = Flask(__name__)

# Function to encode input data using preloaded mappings
def encode_data(input_data):
    # Apply encoding for each feature
    input_data['Gender'] = input_data['Gender'].map(mappings['Gender'])
    input_data['Age'] = input_data['Age'].map(mappings['Age'])
    input_data['Current_Year_1718'] = input_data['Current_Year_1718'].map(mappings['Current_Year_1718'])
    input_data['Proposed_YearGrade_1819'] = input_data['Proposed_YearGrade_1819'].map(mappings['Proposed_YearGrade_1819'])
    input_data['Previous_Curriculum_17182'] = input_data['Previous_Curriculum_17182'].map(mappings['Previous_Curriculum_17182'])
    input_data['Current_School'] = input_data['Current_School'].map(mappings['Current_School'])
    input_data['Current_Curriculum'] = input_data['Current_Curriculum'].map(mappings['Current_Curriculum'])
    input_data['Previous_yearGrade'] = input_data['Previous_yearGrade'].map(mappings['Previous_yearGrade'])
    
    return input_data

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Extract JSON data from the request
    
    # Convert the data into a DataFrame (single row)
    input_data = pd.DataFrame([data])
    
    # Encode the categorical columns in the input data
    input_data = encode_data(input_data)
    
    # Make a prediction using the loaded model
    prediction = svm_model.predict(input_data)
    
    # Return the prediction result as JSON
    return jsonify({'prediction': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
