# file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Preprocessed_Student_Level_Prediction.xlsx'

# %% 
# Step 1: Load and Preprocess the Data (same as before)

import pandas as pd
import numpy as np

df = pd.read_excel('D:/Machine_Learning_Projects/5. Student Level Prediction Using Machine Learning/Preprocessed_Student_Level_Prediction.xlsx')

# Calculate the average of relevant columns (same as before)
columns_to_avg = ['Mathexam', 'Scienceexam_', 'Englishexam_', 'Math191_', 'Science191_', 'English191_',
                  'Math192_', 'Science192_', 'English192_', 'Math193_', 'Science193_', 'English193_',
                  'Math201_', 'Science201_', 'English201_', 'Math202_', 'Science202_', 'English202_',
                  'Math203_', 'Science203_', 'English203_']

df['average'] = df[columns_to_avg].mean(axis=1)

# Create class column based on average score (same as before)
def assign_class(row):
    if row['average'] >= 85:
        return 0  # High average
    elif 75 <= row['average'] < 85:
        return 1  # Medium average
    else:
        return 2  # Low average

df['class'] = df.apply(assign_class, axis=1)

# Drop 'class' and 'average' columns for input features, keep 'class' as target
X = df.drop(columns=['class', 'average'])
y = df['class']

# %% 
# Step 2: K-Fold Cross-Validation Setup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Define the StratifiedKFold cross-validator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 3: Define the Models
# Using the same models as before
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

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

# Step 4: Initialize result dictionary
results = {model_name: {'Accuracy': [], 'F1-Score': [], 'Precision': [], 'Recall': []} for model_name in models}

# Step 5: K-Fold Cross-Validation Loop
for model_name, model in models.items():
    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        # Split data based on current fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store the results for the current fold
        results[model_name]['Accuracy'].append(accuracy)
        results[model_name]['F1-Score'].append(f1)
        results[model_name]['Precision'].append(precision)
        results[model_name]['Recall'].append(recall)
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['High average', 'Medium average', 'Low average'],
                    yticklabels=['High average', 'Medium average', 'Low average'])
        plt.title(f'Confusion Matrix for {model_name} (Fold {fold_num})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# Step 6: Convert results to DataFrame for easier comparison
results_df = pd.DataFrame({
    model_name: {
        'Accuracy': np.mean(result['Accuracy']),
        'F1-Score': np.mean(result['F1-Score']),
        'Precision': np.mean(result['Precision']),
        'Recall': np.mean(result['Recall'])
    }
    for model_name, result in results.items()
}).T

# Step 7: Display results
print("K-Fold Cross-Validation Results")
print(results_df)

correlation_matrix = X.corr()

# Step 8: correlation matrix for all features
# Select the desired categorical columns
import seaborn as sns
import matplotlib.pyplot as plt
selected_columns = [
    'Gender', 
    'Age_as_of_Academic_Year_1718', 
    'Current_Year_1718', 
    'Proposed_YearGrade_1819', 
    'Year_of_Admission', 
    'Previous_Curriculum_17182', 
    'Current_School', 
    'Current_Curriculum', 
    'Previous_yearGrade'
]

# Subset the dataframe to include only these columns
selected_features = df[selected_columns]

# Compute the correlation matrix for the encoded variables
correlation_matrix_selected = selected_features_encoded.corr()

# Plot the heatmap for the selected variables
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Selected Categorical Variables')
plt.show()