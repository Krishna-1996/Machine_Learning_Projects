# %%
# Step 0: import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lime
from lime.lime_tabular import LimeTabularExplainer

# %%
# Step 1: Create a simple synthetic dataset
X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
print("X",X)
print("y",y)
# %%
# Step 2: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Step 3: Train a simple model - Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# %%
# Step 4: Make predictions and evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# %%
# Step 5: Apply LIME to explain a prediction
# Initialize LIME Explainer
explainer = LimeTabularExplainer(X_train, training_labels=y_train, mode="classification", 
                                 feature_names=[f'Feature{i+1}' for i in range(X_train.shape[1])])

# %%
# Select a test instance to explain
instance_to_explain = X_test[0]

# %%
# Get explanation
explanation = explainer.explain_instance(instance_to_explain, model.predict_proba)

# %%
# Step 6: Visualize the explanation
explanation.show_in_notebook(show_table=True, show_all=False)

# %%
# Alternative: Visualizing the explanation as a bar chart
fig = explanation.as_pyplot_figure()
plt.show()
