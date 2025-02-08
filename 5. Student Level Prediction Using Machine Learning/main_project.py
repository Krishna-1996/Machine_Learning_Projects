# %%
# Step 1 Load and Clean Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Student Level Prediction Using Machine Learning.csv'
df = pd.read_csv(file_path)

# Clean column names: remove leading/trailing spaces, replace spaces with underscores, remove non-alphanumeric characters
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

# Filter rows based on curriculum (only American and British)
valid_curricula = ['American', 'British']
df = df[df['Previous_Curriculum_17182'].isin(valid_curricula)]

# Modify 'Year_of_Admission' based on 'Current_School' column
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 1 Current Student':'Current Student'})
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 2 Current Student':'Current Student'})

# Handle missing values: fill categorical with mode, numerical with mean
for col in df.columns:
    if df[col].isnull().sum() > 0:  # If there are null values in the column
        if df[col].dtype == 'object':  # For categorical columns (strings)
            mode_value = df[col].mode()[0]  # Get the most frequent value
            df[col].fillna(mode_value, inplace=True)
        else:  # For numerical columns
            mean_value = df[col].mean()  # Get the mean value
            df[col].fillna(mean_value, inplace=True)

# Step 2: Encode Categorical Data to Numerical
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical to numerical
    label_encoders[col] = le  # Save the encoder for future reference

# Save the preprocessed data and mappings
output_file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Preprocessed_Student_Level_Prediction.xlsx'
mapping_data = []

for col, le in label_encoders.items():
    category_mapping = {index: label for index, label in enumerate(le.classes_)}
    mapping_data.append({"Column Name": col, "Mapping": category_mapping})

mapping_df = pd.DataFrame(mapping_data)

# Write the cleaned data and mapping data to an Excel file
with pd.ExcelWriter(output_file_path) as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    mapping_df.to_excel(writer, sheet_name='Mappings')

print(f"Preprocessing complete. Dataset saved to: {output_file_path}")

# %%
# Step 3: Feature Engineering and Class Assignment
# Load the preprocessed dataset
df = pd.read_excel(output_file_path, sheet_name='Data')

# Calculate the average score for the relevant subjects
columns_to_avg = ['Mathexam', 'Scienceexam_', 'Englishexam_', 'Math191_', 'Science191_', 'English191_',
                  'Math192_', 'Science192_', 'English192_', 'Math193_', 'Science193_', 'English193_',
                  'Math201_', 'Science201_', 'English201_', 'Math202_', 'Science202_', 'English202_',
                  'Math203_', 'Science203_', 'English203_']
df['average'] = df[columns_to_avg].mean(axis=1)

# Assign class based on the average score
def assign_class(row):
    if row['average'] >= 85:
        return 0  # High average
    elif 75 <= row['average'] < 85:
        return 1  # Medium average
    else:
        return 2  # Low average

df['class'] = df.apply(assign_class, axis=1)

# Define input features (X) and target (y)
X = df.drop(columns=['class', 'average'])
y = df['class']

# %%
# Step 4: Model Definition and K-Fold Cross-Validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix

# Define the models to be evaluated
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'ANN (MLP)': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Bernoulli Naive Bayes': BernoulliNB(),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'Stacking': StackingClassifier(estimators=[('rfc', RandomForestClassifier(n_estimators=100, random_state=42)),
                                               ('ann', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)),
                                               ('knn', KNeighborsClassifier(n_neighbors=5))],
                                  final_estimator=LogisticRegression()),
    'Voting Classifier': VotingClassifier(estimators=[('rfc', RandomForestClassifier(n_estimators=100, random_state=42)),
                                                      ('ann', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)),
                                                      ('svm', SVC(kernel='linear', random_state=42)),
                                                      ('knn', KNeighborsClassifier(n_neighbors=5))], voting='hard')
}

# Stratified K-Fold cross-validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize result dictionary
results = {model_name: {'Accuracy': [], 'F1-Score': [], 'Precision': [], 'Recall': []} for model_name in models}

# Loop through each model and perform K-Fold Cross-Validation
for model_name, model in models.items():
    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        # Split data based on the current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        
        # Store the results for the current fold
        results[model_name]['Accuracy'].append(accuracy)
        results[model_name]['F1-Score'].append(f1)
        results[model_name]['Precision'].append(precision)
        results[model_name]['Recall'].append(recall)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['High average', 'Medium average', 'Low average'],
                    yticklabels=['High average', 'Medium average', 'Low average'])
        plt.title(f'Confusion Matrix for {model_name} (Fold {fold_num})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# Convert results to DataFrame for easier comparison
results_df = pd.DataFrame({
    model_name: {
        'Accuracy': np.mean(result['Accuracy']),
        'F1-Score': np.mean(result['F1-Score']),
        'Precision': np.mean(result['Precision']),
        'Recall': np.mean(result['Recall'])
    }
    for model_name, result in results.items()
}).T

# Display the results
print("K-Fold Cross-Validation Results")
print(results_df)

# %%
# Step 5: Correlation Heatmap for Selected Features
selected_columns = [
    'Gender', 'Age_as_of_Academic_Year_1718', 'Current_Year_1718', 
    'Proposed_YearGrade_1819', 'Year_of_Admission', 'Previous_Curriculum_17182', 
    'Current_School', 'Current_Curriculum', 'Previous_yearGrade'
]

# Subset the dataframe to include only the selected columns
selected_features = df[selected_columns]

# Compute the correlation matrix for the selected features
correlation_matrix_selected = selected_features.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Selected Features')
plt.show()
