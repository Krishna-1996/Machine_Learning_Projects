import pandas as pd

# File paths
dataset_file_path = r"D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\final_dataset_file.xlsx"
encoding_file_path = r"D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\Power BI Excel_Update.csv"
output_file_path = r"D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\encoded_final_dataset.xlsx"

# Load the datasets
student_data = pd.read_excel(dataset_file_path)
encoding_data = pd.read_csv(encoding_file_path)

# Create a dictionary to store encoding information
encoding_dict = {}
for _, row in encoding_data.iterrows():
    feature = row['Feature Name']
    value = row['Unique Value']
    num_value = row['Numerical Value']
    
    # Normalize the values to ensure exact match (remove leading/trailing spaces, lowercase)
    value = str(value).strip().lower()  # Normalize value from encoding file
    if feature not in encoding_dict:
        encoding_dict[feature] = {}
    encoding_dict[feature][value] = num_value

# Function to encode categorical columns based on the encoding dictionary
def encode_categorical_columns(df, encoding_dict):
    for column in df.columns:
        if column in encoding_dict:
            # Normalize the values in the dataframe column (strip spaces and lower case)
            df[column] = df[column].apply(lambda x: str(x).strip().lower() if isinstance(x, str) else x)
            
            # Map the categorical values to their numerical equivalents
            df[column] = df[column].map(encoding_dict[column]).fillna(df[column])
    return df

# Encode the categorical features
encoded_student_data = encode_categorical_columns(student_data.copy(), encoding_dict)

# Save the encoded data to a new Excel file
encoded_student_data.to_excel(output_file_path, index=False)

print(f"Encoded data saved to: {output_file_path}")
