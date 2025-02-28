# %%
# Step 1: Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler

# Load the dataset from the provided CSV file
data_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\test_dataset_Loan_Prediction_by_me.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataset to confirm it loaded correctly
print(data.head())

# Convert categorical variables to numeric using LabelEncoder
label_encoders = {}
for column in ['gender', 'income', 'expenses']:  # Assuming these columns are categorical
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Normalize the data
scaler = MinMaxScaler()
data[['income', 'expenses']] = scaler.fit_transform(data[['income', 'expenses']])

# %%
# Step 2: Splitting the Data into Training and Testing

# Features (X) and Target (y)
X = data[['gender', 'income', 'expenses', 'married', 'loan']]
y = data['loan']  # Assuming loan is the target variable

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
user_input = int(input("Enter the UserID (1-40) of the instance to explain: ")) - 1

# Get explanation for that instance
explanation = explainer.explain_instance(X_test.values[user_input], model.predict)
explanation.show_in_notebook()
