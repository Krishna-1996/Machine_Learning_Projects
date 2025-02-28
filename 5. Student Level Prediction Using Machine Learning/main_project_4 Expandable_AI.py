# %%
# Step 1: Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import lime.lime_tabular

# Load the dataset from the provided CSV file
data_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\test_dataset_Loan_Prediction_by_me.csv'
data = pd.read_csv(data_path)

# Check the first few rows of the dataset
print("Columns in the dataset:")
print(data.columns)
print(data.head())

# %%
# Convert categorical variables to numeric using LabelEncoder (gender)
label_encoders = {}
for column in ['gender']:  # Only gender needs encoding
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# %%
# Convert 'income' and 'expenses' into numeric values (since they are categorical ranges)
def income_to_numeric(income_range):
    if '100-200' in income_range:
        return 150
    elif '300-400' in income_range:
        return 350
    elif '201-300' in income_range:
        return 250
    elif '20001-30000' in income_range:
        return 25000
    elif '10000-20000' in income_range:
        return 15000
    else:
        return 0  # Default for missing or unknown ranges

def expenses_to_numeric(expenses_range):
    if '100-200' in expenses_range:
        return 150
    elif '300-400' in expenses_range:
        return 350
    elif '201-300' in expenses_range:
        return 250
    else:
        return 0  # Default for missing or unknown ranges

# Apply these functions to the respective columns
data['income'] = data['income'].apply(income_to_numeric)
data['expenses'] = data['expenses'].apply(expenses_to_numeric)

# %%
# Normalize the data
scaler = MinMaxScaler()
data[['income', 'expenses']] = scaler.fit_transform(data[['income', 'expenses']])

# %%
# Step 2: Splitting the Data into Training and Testing

# Features (X) and Target (y)
X = data[['gender', 'income', 'expenses', 'married']]  # Assuming 'loan' is the target
y = data['loan']  # 'loan' is the target variable

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Step 3: Model Training (Linear Regression)

model = LinearRegression()
model.fit(X_train, y_train)

# %%
# Step 4: Evaluate the Model using Confusion Matrix

# Make predictions
y_pred = model.predict(X_test)

# Threshold predictions to 0 or 1 for confusion matrix
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:")
print(cm)

# %%
# Step 5: Generate Predictions & Save Results to CSV

# Add predictions to the dataset
data['predict_value'] = model.predict(X)
data['True/False'] = np.where(data['loan'] == data['predict_value'], True, False)

# Save the dataframe to CSV
output_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\predictions_output.csv'
data.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")

# %%
# Step 6: LIME (Local Interpretable Model-Agnostic Explanations)

# LIME explainer setup
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns,
    class_names=['0', '1'],
    mode='classification'
)

# User input for which instance to explain
user_input = int(input("Enter the UserID of the instance to explain: ")) - 1

# Get explanation for that instance
explanation = explainer.explain_instance(X_test.values[user_input], model.predict)
explanation.show_in_notebook()
