import pandas as pd

# File paths
dataset_file_path = r"D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\final_dataset_file.xlsx"
encoding_file_path = r"D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\Power BI Excel_Update.csv"
output_file_path = r"D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\encoded_final_dataset.xlsx"

# Load the datasets
student_data = pd.read_excel(dataset_file_path)
encoding_data = pd.read_csv(encoding_file_path)

# Function to encode the categorical columns based on the encoding dictionary
def encode_feature(df, encoding_data):
    for feature in encoding_data['Feature Name'].unique():
        # Get the mapping for the current feature
        feature_data = encoding_data[encoding_data['Feature Name'] == feature][['Unique Value', 'Numerical Value']]
        mapping_dict = dict(zip(feature_data['Numerical Value'], feature_data['Unique Value']))
        
        # If the feature exists in the dataset, apply the mapping
        if feature in df.columns:
            # Replace the numerical values with the corresponding unique values
            df[feature] = df[feature].map(mapping_dict).fillna(df[feature])  # fillna keeps the original value if not in the map
    return df

# Encode the dataset
encoded_student_data = encode_feature(student_data.copy(), encoding_data)

# Save the encoded dataset to a new Excel file
encoded_student_data.to_excel(output_file_path, index=False)

print(f"Encoded data saved to: {output_file_path}")
