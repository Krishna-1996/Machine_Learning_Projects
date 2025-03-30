import pandas as pd

# File paths
csv_file_path = r"D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\final_dataset_file.xlsx"
encoding_file_path = r"D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\Power BI Excel_Update.csv"
output_file_path = r"D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\processed_student_data.xlsx"

# Load the datasets
student_data = pd.read_csv(csv_file_path)
encoding_data = pd.read_excel(encoding_file_path)

# Create a dictionary from the encoding file
encoding_dict = {}
for _, row in encoding_data.iterrows():
    feature = row['Feature Name']
    value = row['Unique Value']
    num_value = row['Numerical Value']
    if feature not in encoding_dict:
        encoding_dict[feature] = {}
    encoding_dict[feature][value] = num_value

# Function to apply encoding to categorical columns
def encode_categorical_columns(df, encoding_dict):
    for column in df.columns:
        if column in encoding_dict:
            # If column has values in the encoding_dict, map them to numerical values
            df[column] = df[column].map(encoding_dict[column]).fillna(df[column])  # Keep non-matching values unchanged
    return df

# Encode categorical columns in student_data
encoded_student_data = encode_categorical_columns(student_data.copy(), encoding_dict)

# Save the new dataframe to Excel
encoded_student_data.to_excel(output_file_path, index=False)

print(f"Data successfully saved to: {output_file_path}")
