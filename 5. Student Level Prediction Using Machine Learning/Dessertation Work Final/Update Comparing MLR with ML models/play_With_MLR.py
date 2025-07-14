# %%
# Step 0: Import necessary files to start
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import matplotlib.pyplot as plt

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# Initialize Excel writer
excel_writer = pd.ExcelWriter("results/results_summary.xlsx", engine="openpyxl")




# %%

# Step 1: Load and Clean Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1.1 Load the dataset
file_path = 'The_Student_Dataset.csv'
df = pd.read_csv(file_path)

# 1.2 Clean column names: remove leading/trailing spaces, replace spaces with underscores, remove non-alphanumeric characters
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove non-alphanumeric characters

# 1.3 Clean unique data: Standardize categories and remove extra spaces
def clean_column_data(column):
    column = column.str.strip()  # Remove leading/trailing spaces
    column = column.replace({
        'Y1': 'Grade 1', 'year1': 'Grade 1', 'grade 1': 'Grade 1', 'Year 1': 'Grade 1',
        'Y2': 'Grade 2', 'year2': 'Grade 2', 'grade 2': 'Grade 2', 'Year 2': 'Grade 2',
        'Y3': 'Grade 3', 'year3': 'Grade 3', 'grade 3': 'Grade 3', 'Year 3': 'Grade 3',
        'Y4': 'Grade 4', 'year3': 'Grade 4', 'grade 4': 'Grade 4', 'Year 4': 'Grade 4',
        'Y5': 'Grade 5', 'year5': 'Grade 5', 'grade 5': 'Grade 5', 'Year 5': 'Grade 5',
        'Y6': 'Grade 6', 'year6': 'Grade 6', 'grade 6': 'Grade 6', 'Grade 6 ': 'Grade 6', 'Year 6': 'Grade 6',
        'Y7': 'Grade 7', 'year7': 'Grade 7', 'grade 7': 'Grade 7', 'Grade 7 ': 'Grade 7', 'Year 7': 'Grade 7',
        'Y8': 'Grade 8', 'year8': 'Grade 8', 'grade 8': 'Grade 8', 'Grade 8 ': 'Grade 8',
        'Y9': 'Grade 9', 'year9': 'Grade 9', 'grade 9': 'Grade 9',
        'Y10': 'Grade 10', 'year10': 'Grade 10', 'grade 10': 'Grade 10', 'Year 10': 'Grade 10',
        'Y11': 'Grade 11', 'year11': 'Grade 11', 'grade 11': 'Grade 11',
        'Y12': 'Grade 12', 'year12': 'Grade 12', 'grade 12': 'Grade 12',
        'Y13': 'Grade 13', 'year13': 'Grade 13', 'grade 13': 'Grade 13',
        'Year System' : 'Year System', 'Year System ' : 'Year System',
        'Grade System' : 'Grade System', 'Grade system' : 'Grade System',
    }, regex=True)
    return column

# 1.4 Apply the cleaning function to all categorical columns
for col in df.columns:
    if df[col].dtype == 'object':  # Only clean non-numeric columns
        df[col] = clean_column_data(df[col])

# 1.5 Filter rows based on curriculum (only American and British)
valid_curricula = ['American', 'British']
df = df[df['Previous_Curriculum_17182'].isin(valid_curricula)]

# 1.6 Modify 'Year_of_Admission' based on 'Current_School' column
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 1 Current Student':'Current Student'})
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 2 Current Student':'Current Student'})





# %%

# Step 2: Handle missing values: fill categorical with mode, numerical with mean
for col in df.columns:
    if df[col].isnull().sum() > 0:  # If there are null values in the column
        if df[col].dtype == 'object':  # For categorical columns (strings)
            mode_value = df[col].mode()[0]  # Get the most frequent value
            df[col].fillna(mode_value, inplace=True)
        else:  # For numerical columns
            mean_value = df[col].mean()  # Get the mean value
            df[col].fillna(mean_value, inplace=True)





# %%

# Step 3: Encode Categorical Data to Numerical
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

# 3.1 Remove rows where 'Year_of_Admission' contains "New Admission 18/19"
df = df[df['Year_of_Admission'] != 'New Admission 18/19']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical to numerical
    label_encoders[col] = le  # Save the encoder for future reference

# Now just drop the column Year_of_Admission.
df.drop(columns=['Year_of_Admission'], inplace=True)
# 3.2 Save the preprocessed data and mappings
output_file_path = 'The_Student_Dataset_Preprocessed.xlsx'
mapping_data = []

for col, le in label_encoders.items():
    category_mapping = {index: label for index, label in enumerate(le.classes_)}
    mapping_data.append({"Column Name": col, "Mapping": category_mapping})

mapping_df = pd.DataFrame(mapping_data)

# 3.3 Write the cleaned data and mapping data to an Excel file
with pd.ExcelWriter(output_file_path) as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    mapping_df.to_excel(writer, sheet_name='Mappings')

print(f"Preprocessing complete. Dataset saved to: {output_file_path}")



# %%

# Step 4: Feature Engineering and Class Assignment
# 4.1 Load the preprocessed dataset
df = pd.read_excel(output_file_path, sheet_name='Data')

# Calculate the average score for the relevant subjects
columns_to_avg = ['Mathexam', 'Scienceexam_', 'Englishexam_', 'Math191_', 'Science191_', 'English191_',
                  'Math192_', 'Science192_', 'English192_', 'Math193_', 'Science193_', 'English193_',
                  'Math201_', 'Science201_', 'English201_', 'Math202_', 'Science202_', 'English202_',
                  'Math203_', 'Science203_', 'English203_']
df['average'] = df[columns_to_avg].mean(axis=1)

# 4.2 Assign class based on the average score
def assign_class(row):
    if row['average'] >= 79.9999:
        return 0  # Above 79.9999
    else:
        return 1  # Below 80

df['class'] = df.apply(assign_class, axis=1)
# 4.3 Save the dataset with the average and class columns to CSV
df.to_csv('results/The_Student_Dataset_Final.csv', index=False)
df.to_excel(excel_writer, sheet_name='Final_Dataset', index=False)

print("Dataset saved to final_dataset_file.csv")

# 4.4 Define input features (X) and target (y)
X = df.drop(columns=['class', 'average'])
y = df['class']

# Step 4.5: Plot the Correlation Heatmap to see feature dependencies
import seaborn as sns
import matplotlib.pyplot as plt

# 4.5.1 Calculate the correlation matrix
correlation_matrix = df.corr()

# 4.5.2 Plot the heatmap
plt.figure(figsize=(30, 25))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features', fontsize=16)
plt.tight_layout()
plt.show()





# %%

# Step 5: Check Imbalance in Features
feature_imbalance = {col: df[col].value_counts(normalize=True) for col in X.columns}

# Create a DataFrame to show imbalance in tabular form
imbalance_df = pd.DataFrame(feature_imbalance)
print("Feature Imbalance (Tabular View):")

# Step 5.1: Check Imbalance in Features
feature_imbalance = {col: df[col].value_counts(normalize=True) for col in X.columns}

# Create a DataFrame to show imbalance in tabular form
imbalance_df = pd.DataFrame(feature_imbalance)

# Save the imbalance DataFrame to an Excel file
imbalance_output_file_path = 'The_Student_Dataset_Feature_Imbalance_Results.xlsx'
imbalance_df.to_excel(excel_writer, sheet_name='Feature_Imbalance')

print(f"Feature imbalance results saved to: {imbalance_output_file_path}")

# %% 
# Step 6: Modify MLR Model to predict 2019 and 2020 average scores
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# %% 
# Step 6.1: Calculate the Average Scores for 2019 and 2020 for each subject

# 2019 Average Scores
df['Math_2019'] = df[['Math191_', 'Math192_', 'Math193_']].mean(axis=1)
df['Science_2019'] = df[['Science191_', 'Science192_', 'Science193_']].mean(axis=1)
df['English_2019'] = df[['English191_', 'English192_', 'English193_']].mean(axis=1)

# 2020 Average Scores
df['Math_2020'] = df[['Math201_', 'Math202_', 'Math203_']].mean(axis=1)
df['Science_2020'] = df[['Science201_', 'Science202_', 'Science203_']].mean(axis=1)
df['English_2020'] = df[['English201_', 'English202_', 'English203_']].mean(axis=1)

# %% 
# Step 6.2: Prepare Data for MLR Model

# Features: Entrance Exam Scores (Math, Science, and English)
X = df[['Mathexam', 'Scienceexam_', 'Englishexam_']]

# Targets for 2019 and 2020
y_2019 = df[['Math_2019', 'Science_2019', 'English_2019']]
y_2020 = df[['Math_2020', 'Science_2020', 'English_2020']]

# %% 
# Step 6.3: Train and Test MLR Model for 2019 scores
# 6.3.1 Split the data into training and testing sets for 2019
X_train_2019, X_test_2019, y_train_2019, y_test_2019 = train_test_split(X, y_2019, test_size=0.2, random_state=42)

# 6.3.2 Initialize and fit the Linear Regression model for 2019
mlr_model_2019 = LinearRegression()
mlr_model_2019.fit(X_train_2019, y_train_2019)

# 6.3.3 Make predictions on the test set for 2019
y_pred_2019 = mlr_model_2019.predict(X_test_2019)

# 6.3.4 Evaluate the 2019 model performance
mse_2019 = mean_squared_error(y_test_2019, y_pred_2019)
r2_2019 = r2_score(y_test_2019, y_pred_2019)

print(f"2019 Model - Mean Squared Error: {mse_2019}")
print(f"2019 Model - R-squared: {r2_2019}")

# Plotting Actual vs Predicted for 2019
plt.figure(figsize=(10, 6))
plt.scatter(y_test_2019['Math_2019'], y_pred_2019[:, 0], color='blue', label='Math')
plt.scatter(y_test_2019['Science_2019'], y_pred_2019[:, 1], color='green', label='Science')
plt.scatter(y_test_2019['English_2019'], y_pred_2019[:, 2], color='red', label='English')
plt.plot([0, 100], [0, 100], 'k--', lw=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores (2019)')
plt.legend()
plt.tight_layout()
plt.show()

# %% 
# Step 6.4: Train and Test MLR Model for 2020 scores

# 6.4.1 Split the data into training and testing sets for 2020
X_train_2020, X_test_2020, y_train_2020, y_test_2020 = train_test_split(X, y_2020, test_size=0.2, random_state=42)

# 6.4.2 Initialize and fit the Linear Regression model for 2020
mlr_model_2020 = LinearRegression()
mlr_model_2020.fit(X_train_2020, y_train_2020)

# 6.4.3 Make predictions on the test set for 2020
y_pred_2020 = mlr_model_2020.predict(X_test_2020)

# 6.4.4 Evaluate the 2020 model performance
mse_2020 = mean_squared_error(y_test_2020, y_pred_2020)
r2_2020 = r2_score(y_test_2020, y_pred_2020)

print(f"2020 Model - Mean Squared Error: {mse_2020}")
print(f"2020 Model - R-squared: {r2_2020}")

# Plotting Actual vs Predicted for 2020
plt.figure(figsize=(10, 6))
plt.scatter(y_test_2020['Math_2020'], y_pred_2020[:, 0], color='blue', label='Math')
plt.scatter(y_test_2020['Science_2020'], y_pred_2020[:, 1], color='green', label='Science')
plt.scatter(y_test_2020['English_2020'], y_pred_2020[:, 2], color='red', label='English')
plt.plot([0, 100], [0, 100], 'k--', lw=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores (2020)')
plt.legend()
plt.tight_layout()
plt.show()

# %% 
# Step 6.5: Evaluate the model for 2019 and 2020

# 6.5.1: Evaluation for 2019

# Calculate error metrics for 2019
mse_2019 = mean_squared_error(y_test_2019, y_pred_2019)
rmse_2019 = mse_2019 ** 0.5
mae_2019 = mean_absolute_error(y_test_2019, y_pred_2019)
r2_2019 = r2_score(y_test_2019, y_pred_2019)

print(f"2019 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_2019}")
print(f"Root Mean Squared Error (RMSE): {rmse_2019}")
print(f"Mean Absolute Error (MAE): {mae_2019}")
print(f"R-squared (R²): {r2_2019}")

# 6.5.2: Evaluation for 2020
mse_2020 = mean_squared_error(y_test_2020, y_pred_2020)
rmse_2020 = mse_2020 ** 0.5
mae_2020 = mean_absolute_error(y_test_2020, y_pred_2020)
r2_2020 = r2_score(y_test_2020, y_pred_2020)

print(f"\n2020 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_2020}")
print(f"Root Mean Squared Error (RMSE): {rmse_2020}")
print(f"Mean Absolute Error (MAE): {mae_2020}")
print(f"R-squared (R²): {r2_2020}")

# Step 6.7: Visualization of Actual vs Predicted Scores

# Plotting Actual vs Predicted for 2019
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test_2019['Math_2019'], y_pred_2019[:, 0], color='blue', label='Math')
plt.scatter(y_test_2019['Science_2019'], y_pred_2019[:, 1], color='green', label='Science')
plt.scatter(y_test_2019['English_2019'], y_pred_2019[:, 2], color='red', label='English')
plt.plot([0, 100], [0, 100], 'k--', lw=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores (2019)')
plt.legend()

# Plotting Actual vs Predicted for 2020
plt.subplot(1, 2, 2)
plt.scatter(y_test_2020['Math_2020'], y_pred_2020[:, 0], color='blue', label='Math')
plt.scatter(y_test_2020['Science_2020'], y_pred_2020[:, 1], color='green', label='Science')
plt.scatter(y_test_2020['English_2020'], y_pred_2020[:, 2], color='red', label='English')
plt.plot([0, 100], [0, 100], 'k--', lw=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores (2020)')
plt.legend()

plt.tight_layout()
plt.show()

# Step 6.8: Residual Plot for 2019 and 2020

# 2019 Residual Plot
residuals_2019 = y_test_2019 - y_pred_2019
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_2019[:, 0], residuals_2019['Math_2019'], color='blue', label='Math')
plt.scatter(y_pred_2019[:, 1], residuals_2019['Science_2019'], color='green', label='Science')
plt.scatter(y_pred_2019[:, 2], residuals_2019['English_2019'], color='red', label='English')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Scores')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted (2019)')
plt.legend()

# 2020 Residual Plot
residuals_2020 = y_test_2020 - y_pred_2020
plt.subplot(1, 2, 2)
plt.scatter(y_pred_2020[:, 0], residuals_2020['Math_2020'], color='blue', label='Math')
plt.scatter(y_pred_2020[:, 1], residuals_2020['Science_2020'], color='green', label='Science')
plt.scatter(y_pred_2020[:, 2], residuals_2020['English_2020'], color='red', label='English')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Scores')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted (2020)')
plt.legend()

plt.tight_layout()
plt.show()

# Step 6.9: Histogram of Residuals

# 2019 Histogram of Residuals
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(residuals_2019['Math_2019'], bins=30, alpha=0.6, color='blue', label='Math')
plt.hist(residuals_2019['Science_2019'], bins=30, alpha=0.6, color='green', label='Science')
plt.hist(residuals_2019['English_2019'], bins=30, alpha=0.6, color='red', label='English')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Histogram (2019)')
plt.legend()

# 2020 Histogram of Residuals
plt.subplot(1, 2, 2)
plt.hist(residuals_2020['Math_2020'], bins=30, alpha=0.6, color='blue', label='Math')
plt.hist(residuals_2020['Science_2020'], bins=30, alpha=0.6, color='green', label='Science')
plt.hist(residuals_2020['English_2020'], bins=30, alpha=0.6, color='red', label='English')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Histogram (2020)')
plt.legend()

plt.tight_layout()
plt.show()

# Step 6.10: Save the Models for Future Predictions
import joblib
joblib.dump(mlr_model_2019, 'results/mlr_model_2019.pkl')
joblib.dump(mlr_model_2020, 'results/mlr_model_2020.pkl')

print("MLR Models for 2019 and 2020 saved successfully!")


# %% 
# %% 
# Step 7: Save the Models for Future Predictions
import joblib
joblib.dump(mlr_model_2019, 'results/mlr_model_2019.pkl')
joblib.dump(mlr_model_2020, 'results/mlr_model_2020.pkl')

print("MLR Models for 2019 and 2020 saved successfully!")






print("======================================================")