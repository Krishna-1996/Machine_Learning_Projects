import pandas as pd

# Load the CSV file
file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Missing values data.csv'
df = pd.read_csv(file_path)

# Display missing values in tabular form
missing_values = df.isnull().sum().reset_index()
missing_values.columns = ['Column Name', 'Missing Values']
missing_values['Percentage'] = (missing_values['Missing Values'] / len(df)) * 100

# Show only columns with missing values
missing_values = missing_values[missing_values['Missing Values'] > 0]

print(missing_values)





# Clean column names
# Remove leading/trailing spaces.
# Replace spaces with underscores.
# Remove non-alphanumeric characters
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove non-alphanumeric characters


# Clean unique data: Standardize categories and remove extra spaces
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

# Apply the cleaning function to all categorical columns
for col in df.columns:
    if df[col].dtype == 'object':  # Only clean non-numeric columns
        df[col] = clean_column_data(df[col])


