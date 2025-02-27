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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# %%
# Step 2: Load the Iris Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# %%
# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# Step 4: Initialize Models
models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# %%
# Step 5: Train Models, Predict and Evaluate Accuracy
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results.append([model_name, accuracy, cm])

# %%
# Step 6: Create DataFrame to Store Results in Tabular Form
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Confusion Matrix"])

# %%
# Step 7: Display the Results in Tabular Form
print(results_df)

# %%
# Step 8: Visualize Model Accuracy Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.show()

# %%
# Step 9: Visualize Confusion Matrix for Each Model
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
# Step 10: Display Classification Report for Each Model
from sklearn.metrics import classification_report

for model_name, model in models.items():
    print(f"\nClassification Report for {model_name}:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
