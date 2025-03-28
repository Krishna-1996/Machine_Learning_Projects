from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import VotingClassifier
import lime.lime_tabular

app = Flask(__name__)

# Load the trained models (assuming you saved them as pickle files)
with open('models/svm_model.pkl', 'rb') as f:
    voting_classifier = pickle.load(f)

# Load LIME explainer for interpretability
def load_lime_explainer():
    # Replace this with actual training data and feature names for the explainer
    X_train = pd.DataFrame(np.random.randn(100, 10))  # Dummy data for explainer
    feature_names = [f"Feature {i}" for i in range(10)]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=['0', '1'],
        mode='classification'
    )
    return explainer

explainer = load_lime_explainer()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input data
        student_data = {
            'Gender': request.form['gender'],
            'Age_as_of_Academic_Year_1718': float(request.form['age_as_of_1718']),
            'Current_Year_1718': int(request.form['current_year_1718']),
            'Proposed_YearGrade_1819': int(request.form['proposed_year_grade_1819']),
            'Previous_Curriculum_17182': request.form['previous_curriculum_17182'],
            'Current_School': request.form['current_school'],
            'Current_Curriculum': request.form['current_curriculum'],
            'Previous_yearGrade': int(request.form['previous_year_grade']),
            'Mathexam': float(request.form['mathexam']),
            'Scienceexam_': float(request.form['scienceexam']),
            'Englishexam_': float(request.form['englishexam']),
            'Math191_': float(request.form['math191']),
            'Science191_': float(request.form['science191']),
            'English191_': float(request.form['english191']),
            'Math192_': float(request.form['math192']),
            'Science192_': float(request.form['science192']),
            'English192_': float(request.form['english192']),
            'Math193_': float(request.form['math193']),
            'Science193_': float(request.form['science193']),
            'English193_': float(request.form['english193']),
            'Math201_': float(request.form['math201']),
            'Science201_': float(request.form['science201']),
            'English201_': float(request.form['english201']),
            'Math202_': float(request.form['math202']),
            'Science202_': float(request.form['science202']),
            'English202_': float(request.form['english202']),
            'Math203_': float(request.form['math203']),
            'Science203_': float(request.form['science203']),
            'English203_': float(request.form['english203']),
        }
        
        user_input = pd.DataFrame([student_data])

        # Predict the grade using the model
        predicted_grade = voting_classifier.predict(user_input)[0]
        chosen_grade = int(request.form['chosen_grade'])

        # Check if the chosen grade is correct
        if predicted_grade == chosen_grade:
            result = "Correct! Your grade prediction is accurate."
            explanation = "The model predicts the correct grade for the student based on the data provided."
        else:
            result = "Incorrect. The chosen grade might not be suitable."
            explanation = "Here’s the model’s suggestion based on the input data. Please provide more accurate data to get a better prediction."

            # Suggest dataset improvements based on LIME explanations
            explanation_details = explainer.explain_instance(user_input.values[0], voting_classifier.predict_proba)
            explanation = explanation + " " + str(explanation_details.as_list())

        return render_template("index.html", result=result, explanation=explanation, student_data=student_data)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
