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


for col in df.columns:
    if df[col].isnull().sum() > 0:  # If there are null values in the column
        if df[col].dtype == 'object':  # For categorical columns (strings)
            mode_value = df[col].mode()[0]  # Get the most frequent value
            df[col].fillna(mode_value, inplace=True)
        else:  # For numerical columns
            mean_value = df[col].mean()  # Get the mean value
            df[col].fillna(mean_value, inplace=True)





            