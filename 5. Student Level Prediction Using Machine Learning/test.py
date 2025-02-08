# %% Step 1: Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# %% Step 2: Load and preprocess the data
# Load dataset from Excel file
df = pd.read_excel('D:/Machine_Learning_Projects/5. Student Level Prediction Using Machine Learning/Preprocessed_Student_Level_Prediction.xlsx')

# Calculate the average of relevant columns for student exam scores
columns_to_avg = ['Mathexam', 'Scienceexam_', 'Englishexam_', 'Math191_', 'Science191_', 'English191_',
                  'Math192_', 'Science192_', 'English192_', 'Math193_', 'Science193_', 'English193_',
                  'Math201_', 'Science201_', 'English201_', 'Math202_', 'Science202_', 'English202_',
                  'Math203_', 'Science203_', 'English203_']
df['average'] = df[columns_to_avg].mean(axis=1)

# Create class column based on average score
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

# %% Step 3: Set up K-Fold Cross-Validation
# Using StratifiedKFold for ensuring that each fold has the same distribution of class labels
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% Step 4: Define the models for classification
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

# %% Step 5: Initialize a dictionary to store the results of each model
# The dictionary will hold accuracy, f1-score, precision, recall, and confusion matrix for each model
results = {model_name: {'Accuracy': [], 'F1-Score': [], 'Precision': [], 'Recall': [], 'Best_Confusion_Matrix': []} for model_name in models}

# %% Step 6: Perform K-Fold Cross-Validation for each model
for model_name, model in models.items():
    best_cm = None
    best_accuracy = 0
    
    # Loop through each fold
    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        # Split data into training and testing sets for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the model on the current fold's training data
        model.fit(X_train, y_train)
        
        # Predict the class labels on the test set
        y_pred = model.predict(X_test)
        
        # Calculate various evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        
        # Store the metrics for this fold
        results[model_name]['Accuracy'].append(accuracy)
        results[model_name]['F1-Score'].append(f1)
        results[model_name]['Precision'].append(precision)
        results[model_name]['Recall'].append(recall)
        
        # If this fold has the best accuracy, store the confusion matrix for this fold
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_cm = confusion_matrix(y_test, y_pred)

    # After looping through all folds, store the best confusion matrix for each model
    results[model_name]['Best_Confusion_Matrix'] = best_cm

# %% Step 7: Display the results of cross-validation in a DataFrame
# Calculate and print the mean of each metric for each model
results_df = pd.DataFrame({
    model_name: {
        'Accuracy': np.mean(result['Accuracy']),
        'F1-Score': np.mean(result['F1-Score']),
        'Precision': np.mean(result['Precision']),
        'Recall': np.mean(result['Recall'])
    }
    for model_name, result in results.items()
}).T

print("K-Fold Cross-Validation Results")
print(results_df)

# %% Step 8: Plot the confusion matrices for each model in a 3x3 grid
# Plot all 9 confusion matrices (one for each model) in a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, (model_name, result) in enumerate(results.items()):
    ax = axes[i]
    cm = result['Best_Confusion_Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, 
                xticklabels=['High average', 'Medium average', 'Low average'],
                yticklabels=['High average', 'Medium average', 'Low average'])
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# %% Step 9: Scatter plots for model evaluation metrics
# Plot scatter plots comparing Accuracy against F1-Score, Precision, and Recall
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Accuracy vs F1-Score
axes[0].scatter(results_df['Accuracy'], results_df['F1-Score'], color='blue')
axes[0].set_title('Accuracy vs F1-Score')
axes[0].set_xlabel('Accuracy')
axes[0].set_ylabel('F1-Score')

# Accuracy vs Precision
axes[1].scatter(results_df['Accuracy'], results_df['Precision'], color='green')
axes[1].set_title('Accuracy vs Precision')
axes[1].set_xlabel('Accuracy')
axes[1].set_ylabel('Precision')

# Accuracy vs Recall
axes[2].scatter(results_df['Accuracy'], results_df['Recall'], color='red')
axes[2].set_title('Accuracy vs Recall')
axes[2].set_xlabel('Accuracy')
axes[2].set_ylabel('Recall')

plt.tight_layout()
plt.show()

