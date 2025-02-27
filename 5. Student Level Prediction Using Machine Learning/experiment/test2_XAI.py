# %%
# Step 1: Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from scipy import stats
import lime
from lime.lime_tabular import LimeTabularExplainer
import shap
from sklearn.inspection import plot_partial_dependence

# %%
# Step 2: Load the Breast Cancer Dataset
# Load the dataset from sklearn
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Convert to DataFrame for easier exploration
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y


# %%
# Step 3: Data Exploration

# Check for missing values
print("\nMissing values in dataset:\n", df.isnull().sum())

# Display basic statistics
print("\nDataset Statistics:\n", df.describe())

# Visualize the correlation between features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# %%
# Step 4: Outlier Detection and Treatment

# 4.1: Visualizing Outliers using Boxplots
plt.figure(figsize=(15, 10))
df_features = df.drop(columns=['target'])
for i, feature in enumerate(df_features.columns):
    plt.subplot(5, 6, i+1)
    sns.boxplot(df_features[feature])
    plt.title(f'Boxplot: {feature}')
plt.tight_layout()
plt.show()

# 4.2: Z-Score Method for Outlier Detection
z_scores = np.abs(stats.zscore(df_features))
outliers = (z_scores > 3)  # Any z-score greater than 3 is considered an outlier
outliers_count = np.sum(outliers, axis=0)

print("\nOutliers detected for each feature (z-score > 3):")
for feature, count in zip(df_features.columns, outliers_count):
    print(f"{feature}: {count} outliers")

# 4.3: Remove Outliers (optional)
df_no_outliers = df[~(outliers.any(axis=1))]

# %%
# Step 5: Check for Class Imbalance
class_counts = df['target'].value_counts()
print("\nClass Distribution:\n", class_counts)

# Plot class distribution
sns.countplot(x='target', data=df)
plt.title('Class Distribution (0: Malignant, 1: Benign)')
plt.show()

# If there's a class imbalance, use SMOTE (Synthetic Minority Over-sampling Technique)
# or down-sampling the majority class (here we will use down-sampling for simplicity)

if class_counts[0] > class_counts[1]:
    # Down-sample the majority class
    df_minority = df[df['target'] == 1]
    df_majority = df[df['target'] == 0]
    
    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # Don't sample with replacement
                                       n_samples=len(df_minority),  # Equal samples as minority
                                       random_state=42)
    
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
else:
    df_balanced = df

# %%
# Step 6: Check for Noise
# Noise can be detected using methods like variance threshold or correlation-based filtering.

# 6.1: Remove low variance features (noise)
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)  # Threshold can be adjusted
X_filtered = selector.fit_transform(df_balanced.drop(columns=['target']))

# Get the list of features that remain after variance thresholding
features_selected = df_balanced.drop(columns=['target']).columns[selector.get_support()]
print("\nFeatures selected after variance thresholding:", features_selected)

# %%
# Step 7: Preprocessing - Handle Missing Values and Standardize Features

# 7.1: Impute missing values (if any)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_balanced.drop(columns=['target'])), columns=df_balanced.drop(columns=['target']).columns)
df_imputed['target'] = df_balanced['target']

# 7.2: Scale Features (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_imputed.drop(columns=['target']))
y_scaled = df_imputed['target']

# %%
# Step 8: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# %%
# Step 9: Initialize and Train Models
models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Train and evaluate each model (use previously processed data)
results = []


# Train and Evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results.append([model_name, accuracy, cm])

# %%
# Step 10: Display Results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Confusion Matrix"])

# Display accuracy comparison
print(results_df)

# %%
# Step 11: Visualize Model Accuracy Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.show()

# %%
# Step 12: Visualize Confusion Matrix for Each Model
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (model_name, _, cm) in enumerate(results):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
    axes[idx].set_title(model_name)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# %%
# Step 13: Classification Report for Each Model
for model_name, model in models.items():
    print(f"\nClassification Report for {model_name}:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# %%
# Step 14: Store Results for XAI Techniques
# Initialize dictionaries to store XAI results for each model
lime_results = []
shap_results = []
pdp_results = []

# %%
# Step 15: LIME (Local Interpretable Model-Agnostic Explanations)
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    # Initialize the LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=df_imputed.drop(columns=['target']).columns,
        class_names=['Malignant', 'Benign'],
        mode='classification',
        discretize_continuous=True
    )
    
    # Pick an instance from the test set
    instance = X_test[1]
    
    # Explain the model's prediction for this instance
    explanation = explainer.explain_instance(instance, model.predict_proba)
    
    # Save explanation to results
    lime_results.append([model_name, explanation.as_list()])

# Step 16: SHAP (Shapley Additive Explanations)
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    explainer_shap = shap.TreeExplainer(model)  # SHAP is often used for tree models
    shap_values = explainer_shap.shap_values(X_test)
    
    # Save explanation to results
    shap_results.append([model_name, shap_values[1]])


# Step 17: Partial Dependence Plots (PDPs)
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    features_to_plot = [0, 1, 2, 3]  # Example features
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_partial_dependence(model, X_train, features_to_plot,
                            feature_names=df_imputed.drop(columns=['target']).columns,
                            ax=ax, grid_resolution=50)
    pdp_results.append([model_name, ax])

# %%
# Step 18: Store Results in CSV
# Convert results to DataFrames
lime_df = pd.DataFrame(lime_results, columns=["Model", "LIME Explanation"])
shap_df = pd.DataFrame(shap_results, columns=["Model", "SHAP Values"])
pdp_df = pd.DataFrame(pdp_results, columns=["Model", "PDP Plot"])

# Combine results into one DataFrame for comparison
final_results_df = pd.concat([lime_df, shap_df, pdp_df], axis=1)

# Save results to CSV (including a final comparison sheet)
with pd.ExcelWriter('XAI_Results.xlsx') as writer:
    lime_df.to_excel(writer, sheet_name='LIME Results')
    shap_df.to_excel(writer, sheet_name='SHAP Results')
    pdp_df.to_excel(writer, sheet_name='PDP Results')
    final_results_df.to_excel(writer, sheet_name='Final Comparison')

print("Results saved to XAI_Results.xlsx")