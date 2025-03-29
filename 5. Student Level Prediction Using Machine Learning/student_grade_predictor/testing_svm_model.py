import pandas as pd
import numpy as np
import random
import joblib

# Load the trained model (replace 'model.pkl' with your actual model file path)
# model = joblib.load(r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\The_SVM_Model_Output.pkl')
model = joblib.load(r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\svm_model.pkl')

# Feature encoding information
gender_values = [0, 1]  # Female = 0, Male = 1
age_values = list(range(3, 18))  # Age between 3 and 17
current_year_values = list(range(0, 15))  # Categories for "Current_Year_1718"
proposed_year_values = list(range(0, 14))  # Categories for "Proposed_YearGrade_1819"
previous_curriculum_values = [0, 1]  # American = 0, British = 1
current_school_values = [0, 1]  # School 1 = 0, School 2 = 1
current_curriculum_values = [0, 1]  # American = 0, British = 1
previous_year_grade_values = [0, 1]  # Grade System = 0, Year System = 1
# Assuming exam scores range from 55 to 100 for simplicity
exam_score_range = (55, 100)

# Generate random test cases
def generate_random_test_case():
    test_case = {
        "Gender": random.choice(gender_values),
        "Age_as_of_Academic_Year_1718": random.choice(age_values),
        "Current_Year_1718": random.choice(current_year_values),
        "Proposed_YearGrade_1819": random.choice(proposed_year_values),
        "Previous_Curriculum_17182": random.choice(previous_curriculum_values),
        "Current_School": random.choice(current_school_values),
        "Current_Curriculum": random.choice(current_curriculum_values),
        "Previous_yearGrade": random.choice(previous_year_grade_values),
        "Mathexam": random.randint(*exam_score_range),
        "Scienceexam_": random.randint(*exam_score_range),
        "Englishexam_": random.randint(*exam_score_range),
        "Math191_": random.randint(*exam_score_range),
        "Science191_": random.randint(*exam_score_range),
        "English191_": random.randint(*exam_score_range),
        "Math192_": random.randint(*exam_score_range),
        "Science192_": random.randint(*exam_score_range),
        "English192_": random.randint(*exam_score_range),
        "Math193_": random.randint(*exam_score_range),
        "Science193_": random.randint(*exam_score_range),
        "English193_": random.randint(*exam_score_range),
        "Math201_": random.randint(*exam_score_range),
        "Science201_": random.randint(*exam_score_range),
        "English201_": random.randint(*exam_score_range),
        "Math202_": random.randint(*exam_score_range),
        "Science202_": random.randint(*exam_score_range),
        "English202_": random.randint(*exam_score_range),
        "Math203_": random.randint(*exam_score_range),
        "Science203_": random.randint(*exam_score_range),
        "English203_": random.randint(*exam_score_range),
    }
    return test_case

# Run tests and collect results
num_tests = 500  # You can change this to 400, 500, or more
results = []

for i in range(num_tests):
    test_case = generate_random_test_case()
    # Convert the test case into a DataFrame
    test_case_df = pd.DataFrame([test_case])
    
    # Make the prediction
    prediction = model.predict(test_case_df)
    
    # Record the result
    results.append({
        **test_case,  # All feature values
        "Prediction": prediction[0]  # Model prediction (0 or 1)
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to Excel
results_df.to_excel("model_predictions.xlsx", index=False)

# Print summary
prediction_counts = results_df["Prediction"].value_counts()
print(f"Predictions Summary:\n{prediction_counts}")
