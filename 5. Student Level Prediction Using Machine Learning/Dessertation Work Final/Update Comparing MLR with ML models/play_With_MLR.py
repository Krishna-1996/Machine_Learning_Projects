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
df.drop(columns=['Age as of Academic Year 17/18','Current Year (17/18)',
                 'Proposed Year/Grade (18/19)','Current School ','Current Curriculum ','Previous year/Grade ',
                 'Gender','Previous Curriculum (17/18)2'], inplace=True)

# 1.2 Clean column names: remove leading/trailing spaces, replace spaces with underscores, remove non-alphanumeric characters
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove non-alphanumeric characters


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

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical to numerical
    label_encoders[col] = le  # Save the encoder for future reference

# 3.1 Save the preprocessed data and mappings
output_file_path = 'The_Student_Dataset_Preprocessed.xlsx'
mapping_data = []

for col, le in label_encoders.items():
    category_mapping = {index: label for index, label in enumerate(le.classes_)}
    mapping_data.append({"Column Name": col, "Mapping": category_mapping})

mapping_df = pd.DataFrame(mapping_data)

# 3.2 Write the cleaned data and mapping data to an Excel file
with pd.ExcelWriter(output_file_path) as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    mapping_df.to_excel(writer, sheet_name='Mappings')

print(f"Preprocessing complete. Dataset saved to: {output_file_path}")



# %%
# Step 4: Calculate average for each subject in 2019
df['Math_2019'] = df[['Math191_', 'Math192_', 'Math193_']].mean(axis=1)
df['Science_2019'] = df[['Science191_', 'Science192_', 'Science193_']].mean(axis=1)
df['English_2019'] = df[['English191_', 'English192_', 'English193_']].mean(axis=1)

# %%
# Step 5: Investigate correlation between entrance exam score and 2019 average score
# 5.1 For Math
math_corr = df[['Mathexam', 'Math_2019']].corr().iloc[0, 1]

# 5.2 For Science
science_corr = df[['Scienceexam_', 'Science_2019']].corr().iloc[0, 1]

# 5.3 For English
english_corr = df[['Englishexam_', 'English_2019']].corr().iloc[0, 1]

# Print the correlation coefficients
print(f"Correlation between Entrance Math Exam and 2019 Math Average: {math_corr}")
print(f"Correlation between Entrance Science Exam and 2019 Science Average: {science_corr}")
print(f"Correlation between Entrance English Exam and 2019 English Average: {english_corr}")

# %%
# Step 6: Visualize the data (optional but useful)
# 6.1 Scatter plot for Math
plt.figure(figsize=(10, 6))
plt.scatter(df['Mathexam'], df['Math_2019'], alpha=0.5)
plt.title('Entrance Math Exam Score vs 2019 Math Average')
plt.xlabel('Mathexam')
plt.ylabel('Math_2019 Average')
plt.show()

# 6.2 Scatter plot for Science
plt.figure(figsize=(10, 6))
plt.scatter(df['Scienceexam_'], df['Science_2019'], alpha=0.5)
plt.title('Entrance Science Exam Score vs 2019 Science Average')
plt.xlabel('Scienceexam_')
plt.ylabel('Science_2019 Average')
plt.show()

# 6.3 Scatter plot for English
plt.figure(figsize=(10, 6))
plt.scatter(df['Englishexam_'], df['English_2019'], alpha=0.5)
plt.title('Entrance English Exam Score vs 2019 English Average')
plt.xlabel('Englishexam_')
plt.ylabel('English_2019 Average')
plt.show()
