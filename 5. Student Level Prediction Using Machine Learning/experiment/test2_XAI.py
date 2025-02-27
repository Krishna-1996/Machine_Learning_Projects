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
from sklearn.inspection import PartialDependenceDisplay


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
# Step 3: Preprocessing
# Handling missing values, outliers, and scaling as done previously
# (similar steps from previous code)

# %%
# Step 4: Models Setup
models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Train and evaluate each model (use previously processed data)
results = []

# Step 5: Store Results for XAI Techniques
# Initialize dictionaries to store XAI results for each model
lime_results = []
shap_results = []
pdp_results = []

# %%
# Step 14: LIME (Local Interpretable Model-Agnostic Explanations)
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

# Step 15: SHAP (Shapley Additive Explanations)
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    explainer_shap = shap.TreeExplainer(model)  # SHAP is often used for tree models
    shap_values = explainer_shap.shap_values(X_test)
    
    # Save explanation to results
    shap_results.append([model_name, shap_values[1]])

# Step 16: Integrated Gradients (Not applicable for tree models but will include as a placeholder)
# Integrated Gradients is typically used for neural networks, so we'll skip it here, but for completeness:
# We could add integrated gradients for deep learning models if needed.

# Step 17: Grad-CAM (Gradient-weighted Class Activation Mapping)
# Grad-CAM is used for deep learning models (especially CNNs). So, it's not applicable for tree models.
# Grad-CAM is used primarily for image classification tasks, where convolutional layers are present.
# We'll skip this for tree-based models.

# Step 18: Partial Dependence Plots (PDPs)
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    features_to_plot = [0, 1, 2, 3]  # Example features
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_partial_dependence(model, X_train, features_to_plot,
                            feature_names=df_imputed.drop(columns=['target']).columns,
                            ax=ax, grid_resolution=50)
    pdp_results.append([model_name, ax])

# %%
# Step 19: Store Results in CSV
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
