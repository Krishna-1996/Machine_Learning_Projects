# Preprocessing and SVM Model Training
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1.1 Load the dataset
file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Student Level Prediction Using Machine Learning - Copy.csv'  # Your dataset path
df = pd.read_csv(file_path)

# 1.2 Clean column names
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove non-alphanumeric characters

# 1.3 Clean unique data
def clean_column_data(column):
    column = column.str.strip()
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

# 1.4 Apply the cleaning function
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = clean_column_data(df[col])

# 1.5 Filter rows based on curriculum
valid_curricula = ['American', 'British']
df = df[df['Previous_Curriculum_17182'].isin(valid_curricula)]

# 1.6 Modify 'Year_of_Admission'
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 1 Current Student':'Current Student'})
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 2 Current Student':'Current Student'})

# Handle missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
        else:
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)

# Encode Categorical Data
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

# Remove rows where 'Year_of_Admission' contains "New Admission 18/19"
df = df[df['Year_of_Admission'] != 'New Admission 18/19']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical to numerical
    label_encoders[col] = le  # Save the encoder for future reference

df.drop(columns=['Year_of_Admission'], inplace=True)

# Feature Engineering
columns_to_avg = ['Mathexam', 'Scienceexam_', 'Englishexam_', 'Math191_', 'Science191_', 'English191_',
                  'Math192_', 'Science192_', 'English192_', 'Math193_', 'Science193_', 'English193_',
                  'Math201_', 'Science201_', 'English201_', 'Math202_', 'Science202_', 'English202_',
                  'Math203_', 'Science203_', 'English203_']
df['average'] = df[columns_to_avg].mean(axis=1)

def assign_class(row):
    return 0 if row['average'] >= 80 else 1

df['class'] = df.apply(assign_class, axis=1)

# Define input features and target
X = df.drop(columns=['class', 'average'])
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(probability=True, kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred_svm = svm_model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred_svm)
accuracy = accuracy_score(y_test, y_pred_svm)

print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix for SVM Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the model
svm_model_save_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\The_SVM_Model_Output.pkl'  # Path to save the model
joblib.dump(svm_model, svm_model_save_path)
print(f"SVM model saved to: {svm_model_save_path}")

# Save the preprocessed data and mappings
output_file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\student_grade_predictor\Only_SVM\Preprocessed_Student_Level_Prediction.xlsx'  # Path to save the preprocessed data

mapping_data = []
for col, le in label_encoders.items():
    category_mapping = {index: label for index, label in enumerate(le.classes_)}
    mapping_data.append({"Column Name": col, "Mapping": category_mapping})

mapping_df = pd.DataFrame(mapping_data)

with pd.ExcelWriter(output_file_path) as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    mapping_df.to_excel(writer, sheet_name='Mappings')

print(f"Preprocessing complete. Dataset saved to: {output_file_path}")
