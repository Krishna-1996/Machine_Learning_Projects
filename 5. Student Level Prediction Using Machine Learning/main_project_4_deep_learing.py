# %% 
# Importing Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

# %% Step 1: Data Preprocessing (same as you provided)
# 1.1 Load the dataset
file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Student Level Prediction Using Machine Learning.csv'
df = pd.read_csv(file_path)

# 1.2 Clean column names: remove leading/trailing spaces, replace spaces with underscores, remove non-alphanumeric characters
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove non-alphanumeric characters

# 1.3 Clean unique data: Standardize categories and remove extra spaces
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

# 1.4 Apply the cleaning function to all categorical columns
for col in df.columns:
    if df[col].dtype == 'object':  # Only clean non-numeric columns
        df[col] = clean_column_data(df[col])

# 1.5 Filter rows based on curriculum (only American and British)
valid_curricula = ['American', 'British']
df = df[df['Previous_Curriculum_17182'].isin(valid_curricula)]

# 1.6 Modify 'Year_of_Admission' based on 'Current_School' column
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 1 Current Student':'Current Student'})
df['Year_of_Admission'] = df['Year_of_Admission'].replace({'School 2 Current Student':'Current Student'})

# %%
# Step 1.2: Handle missing values: fill categorical with mode, numerical with mean
for col in df.columns:
    if df[col].isnull().sum() > 0:  # If there are null values in the column
        if df[col].dtype == 'object':  # For categorical columns (strings)
            mode_value = df[col].mode()[0]  # Get the most frequent value
            df[col].fillna(mode_value, inplace=True)
        else:  # For numerical columns
            mean_value = df[col].mean()  # Get the mean value
            df[col].fillna(mean_value, inplace=True)

# %%
# Step 1.3: Encode Categorical Data to Numerical
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

# 3.1 Remove rows where 'Year_of_Admission' contains "New Admission 18/19"
df = df[df['Year_of_Admission'] != 'New Admission 18/19']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical to numerical
    label_encoders[col] = le  # Save the encoder for future reference

# Now just drop the column Year_of_Admission.
df.drop(columns=['Year_of_Admission'], inplace=True)
# 3.2 Save the preprocessed data and mappings
output_file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Preprocessed_Student_Level_Prediction.xlsx'
mapping_data = []

for col, le in label_encoders.items():
    category_mapping = {index: label for index, label in enumerate(le.classes_)}
    mapping_data.append({"Column Name": col, "Mapping": category_mapping})

mapping_df = pd.DataFrame(mapping_data)

# 3.3 Write the cleaned data and mapping data to an Excel file
with pd.ExcelWriter(output_file_path) as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    mapping_df.to_excel(writer, sheet_name='Mappings')

print(f"Preprocessing complete. Dataset saved to: {output_file_path}")

# %%
# Step 1.4: Feature Engineering and Class Assignment
# 4.1 Load the preprocessed dataset
df = pd.read_excel(output_file_path, sheet_name='Data')

# Calculate the average score for the relevant subjects
columns_to_avg = ['Mathexam', 'Scienceexam_', 'Englishexam_', 'Math191_', 'Science191_', 'English191_',
                  'Math192_', 'Science192_', 'English192_', 'Math193_', 'Science193_', 'English193_',
                  'Math201_', 'Science201_', 'English201_', 'Math202_', 'Science202_', 'English202_',
                  'Math203_', 'Science203_', 'English203_']
df['average'] = df[columns_to_avg].mean(axis=1)

# 4.2 Assign class based on the average score
def assign_class(row):
    if row['average'] >= 79.9999:
        return 0  # Above 79.9999
    else:
        return 1  # Below 80

df['class'] = df.apply(assign_class, axis=1)
# 4.3 Save the dataset with the average and class columns to CSV
df.to_csv('final_dataset_file.csv', index=False)

print("Dataset saved to final_dataset_file.csv")

# 4.4 Define input features (X) and target (y)
X = df.drop(columns=['class', 'average'])
y = df['class']

# %%
# Step 1.5: Check Imbalance in Features
feature_imbalance = {col: df[col].value_counts(normalize=True) for col in X.columns}

# Create a DataFrame to show imbalance in tabular form
imbalance_df = pd.DataFrame(feature_imbalance)
print("Feature Imbalance (Tabular View):")

# Step 5: Check Imbalance in Features
feature_imbalance = {col: df[col].value_counts(normalize=True) for col in X.columns}

# Create a DataFrame to show imbalance in tabular form
imbalance_df = pd.DataFrame(feature_imbalance)

# Save the imbalance DataFrame to an Excel file
imbalance_output_file_path = r'D:\Machine_Learning_Projects\5. Student Level Prediction Using Machine Learning\Feature_Imbalance_Results.xlsx'
imbalance_df.to_excel(imbalance_output_file_path, index=True)

print(f"Feature imbalance results saved to: {imbalance_output_file_path}")


# %% Step 2: 
# Here we will create a simple deep learning model (a neural network)
def create_deep_learning_model(input_dim):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_dim=input_dim))  # Input layer
    
    # Adding hidden layers
    model.add(layers.Dense(128, activation='relu'))  # First hidden layer
    model.add(layers.Dropout(0.3))  # Dropout for regularization
    
    model.add(layers.Dense(64, activation='relu'))  # Second hidden layer
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(32, activation='relu'))  # Third hidden layer
    model.add(layers.Dropout(0.3))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# %% Step 3: Train the Deep Learning Model using K-Fold Cross Validation

# Convert your features (X) and target (y) into NumPy arrays for deep learning
X = df.drop(columns=['class', 'average']).values
y = df['class'].values

# Define the k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results for deep learning model
dl_results = {'Accuracy': [], 'F1-Score': [], 'Precision': [], 'Recall': [], 'ROC AUC': []}
dl_confusion_matrices = {}

for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
    print(f"Training Fold {fold_num} of Deep Learning Model...")
    
    # Split the data based on the current fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Create and train the deep learning model
    model = create_deep_learning_model(input_dim=X_train.shape[1])
    
    # Fit the model (with validation split)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    
    # Make predictions on the test data
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    dl_results['Accuracy'].append(accuracy)
    dl_results['F1-Score'].append(f1)
    dl_results['Precision'].append(precision)
    dl_results['Recall'].append(recall)
    dl_results['ROC AUC'].append(roc_auc)
    
    # Store confusion matrix for each fold
    cm = confusion_matrix(y_test, y_pred)
    dl_confusion_matrices[f"Fold {fold_num}"] = cm

# %% Step 4: Evaluate the Deep Learning Model

# Calculate average metrics for the deep learning model
avg_dl_results = {metric: np.mean(dl_results[metric]) for metric in dl_results}
print("Deep Learning Model Evaluation Results (Average across all folds):")
print(pd.DataFrame(avg_dl_results, index=[0]))

# %% Step 5: Visualize Deep Learning Model Performance

# Plot the training/validation loss curves
plt.figure(figsize=(10, 6))
for fold_num, cm in dl_confusion_matrices.items():
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'])
    plt.title(f"Confusion Matrix - {fold_num}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plot the ROC curve for Deep Learning
plt.figure(figsize=(10, 8))
for fold_num, cm in dl_confusion_matrices.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test).ravel())
    plt.plot(fpr, tpr, label=f"Fold {fold_num} (AUC = {roc_auc_score(y_test, y_pred):.2f})")
    
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %% Step 6: Display Keras Model Training History (for loss curves)

# Optionally, plot the training history of the deep learning model to show training/validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% Step 7: Compare Deep Learning Model with Classical Models
# You can compare the deep learning model results with the classical models (e.g., Random Forest, XGBoost, etc.)
# Use the `metrics_df` generated earlier for comparison with the deep learning results.

# Add the Deep Learning Results to the Comparison Table
dl_results_df = pd.DataFrame(dl_results)
dl_results_df = dl_results_df.mean(axis=0).round(3)

# Print comparison
print("\nClassical Models vs Deep Learning Model Comparison (Averages):")
print(pd.concat([metrics_df, dl_results_df.T], axis=1, keys=["Classical Models", "Deep Learning"]))

