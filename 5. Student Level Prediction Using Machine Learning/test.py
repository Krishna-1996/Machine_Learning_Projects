import pandas as pd

# Load the dataset
file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Student Level Prediction Using Machine Learning.csv'
df = pd.read_csv(file_path)
print(df.columns)


# Step 1: Clean columns name
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from columns name
df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove any non-alphanumeric characters

# Step 2: Clean unique data (Standardize categories and remove extra spaces)
def clean_column_data(column):
    # Replace known variants (e.g., "Y1", "year1", "grade 1" -> "Grade 1")
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
           
        
        # Add more replacements as necessary for other columns
    }, regex=True)
    return column

# Apply the cleaning function to all columns with categorical data
for col in df.columns:
    if df[col].dtype == 'object':  # Only clean non-numeric columns (strings)
        df[col] = clean_column_data(df[col])

# Step 3: Data Filtering based on 'Previous_Curriculum_17182' column
valid_curricula = ['American', 'British']
df = df[df['Previous_Curriculum_17182'].isin(valid_curricula)]  # Keep only rows where curriculum is American or British

# Step 4: Modify the 'Year_of_Admission' column based on 'Current_School' column
# If the 'Current_School' is 'School 1' or 'School 2', update 'Year_of_Admission' to 'Current Student'
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 1 Current Student':'Current Student'})
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 2 Current Student':'Current Student'})

# Step 5: Check and handle null values in each column
null_counts = df.isnull().sum()

# Print the count of null values per feature
# print("Null values in each feature before handling:")
# print(null_counts)

# Handle null values
for col in df.columns:
    if df[col].isnull().sum() > 0:  # If there are null values in the column
        if df[col].dtype == 'object':  # For categorical columns (strings)
            # Fill missing values with the mode (most frequent value)
            mode_value = df[col].mode()[0]  # Get the most frequent value
            df[col].fillna(mode_value, inplace=True)
        else:  # For numerical columns
            # Fill missing values with the mean (for numerical columns)
            mean_value = df[col].mean()  # Get the mean value
            df[col].fillna(mean_value, inplace=True)

# Check for null values after handling
null_counts_after = df.isnull().sum()

# Print the count of null values after handling
# print("\nNull values in each feature after handling:")
# print(null_counts_after)

print(df.columns)

# Step 6: Convert categorical data into numerical data (Mapping values)
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns (object type)
categorical_columns = df.select_dtypes(include=['object']).columns

# Apply Label Encoding to all categorical columns
label_encoders = {}  # Store encoders for reference
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical to numerical
    label_encoders[col] = le  # Save the encoder for future reference

# Save the mapping table for label encoding
mapping_data = []
for col, le in label_encoders.items():
    category_mapping = {index: label for index, label in enumerate(le.classes_)}
    mapping_data.append({"Column Name": col, "Mapping": category_mapping})

# Convert mapping data to DataFrame
mapping_df = pd.DataFrame(mapping_data)

# Save the updated dataset and mappings
output_file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Preprocessed_Student_Level_Prediction.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    mapping_df.to_excel(writer, sheet_name='Mappings')

print(f"All categorical features are now numerical. Dataset saved to: {output_file_path}")
