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
# Mappings for Current_Year_1718 and Proposed_YearGrade_1819 columns
year_mapping = {
    'KG1': 14, 'KG2': 15, 'FS1': 16, 'FS2': 17,
    'Grade 1': 1, 'Grade 2': 2, 'Grade 3': 3, 'Grade 4': 4,
    'Grade 5': 5, 'Grade 6': 6, 'Grade 7': 7, 'Grade 8': 8,
    'Grade 9': 9, 'Grade 10': 10, 'Grade 11': 11, 'Grade 12': 12,
    'Grade 13': 13
}

# Mappings for Previous_yearGrade column
previous_year_mapping = {
    'Grade System': 0,
    'Year System': 1
}

# Ensure categorical values are stripped of spaces before mapping
df['Current_Year_1718'] = df['Current_Year_1718'].str.strip().map(year_mapping)
df['Proposed_YearGrade_1819'] = df['Proposed_YearGrade_1819'].str.strip().map(year_mapping)
df['Previous_yearGrade'] = df['Previous_yearGrade'].str.strip().map(previous_year_mapping)

# Fill any unmapped values with a default value (e.g., -1 for unknown categories)
df['Current_Year_1718'].fillna(-1, inplace=True)
df['Proposed_YearGrade_1819'].fillna(-1, inplace=True)
df['Previous_yearGrade'].fillna(-1, inplace=True)

# Step 7: Save the cleaned and encoded dataframe to a new CSV file
output_file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Cleaned_Student_Level_Prediction_Encoded.csv'
df.to_csv(output_file_path, index=False)

# Step 8: Save the mapping table in a separate sheet (using Excel writer)
mapping_data = {
    "Column Name": ['Current_Year_1718', 'Proposed_YearGrade_1819', 'Previous_yearGrade'],
    "Category": [
        'KG1, KG2, FS1, FS2, Grade 1, Grade 2, Grade 3, Grade 4, Grade 5, Grade 6, Grade 7, Grade 8, Grade 9, Grade 10, Grade 11, Grade 12,Grade 13',
        'KG1, KG2, FS1, FS2, Grade 1, Grade 2, Grade 3, Grade 4, Grade 5, Grade 6, Grade 7, Grade 8, Grade 9, Grade 10, Grade 11, Grade 12,Grade 13',
        'Grade System, Year System'
    ],
    "Mapped Value": [
        '14=KG1, 15=KG2, 16=FS1, 17=FS2, 1=Grade 1, 2=Grade 2, 3=Grade 3, 4=Grade 4, 5=Grade 5, 6=Grade 6, 7=Grade 7, 8=Grade 8, 9=Grade 9, 10=Grade 10, 11=Grade 11, 12=Grade 12, 13=Grade 13',
        '14=KG1, 15=KG2, 16=FS1, 17=FS2, 1=Grade 1, 2=Grade 2, 3=Grade 3, 4=Grade 4, 5=Grade 5, 6=Grade 6, 7=Grade 7, 8=Grade 8, 9=Grade 9, 10=Grade 10, 11=Grade 11, 12=Grade 12, 13=Grade 13',
        '0=Grade System, 1=Year System'
    ]
}

# Convert the mapping information to a DataFrame
mapping_df = pd.DataFrame(mapping_data)

# Save the mapping table to the second sheet in the Excel file
with pd.ExcelWriter(output_file_path.replace('.csv', '.xlsx')) as writer:
    df.to_excel(writer, sheet_name='Data')
    mapping_df.to_excel(writer, sheet_name='Mappings')

print(f"Cleaning, encoding, and saving the dataset complete. The cleaned and encoded dataset is saved to: {output_file_path.replace('.csv', '.xlsx')}")
# Load the dataset
file_path = 'D:/Machine_Learning_Projects/5. Student Level Prediction Using Machine Learning/Cleaned_Student_Level_Prediction.csv'
df2 = pd.read_csv(file_path)
print(df[['Current_Year_1718', 'Proposed_YearGrade_1819', 'Previous_yearGrade']].head(10))
print(df2.head)