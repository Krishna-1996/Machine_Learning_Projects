import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Updated Data (10 rows, 5 features)
data = {
    'Height': [160, 170, 180, 150, 165, 175, 160, 162, 168, 178],
    'Weight': [55, 60, 70, 50, 57, 65, 52, 56, 63, 68],
    'Age': [25, 30, 35, 22, 28, 32, 24, 27, 29, 31],
    'Income': [50, 60, 70, 40, 55, 65, 48, 53, 58, 62],  # In $1000
    'Education Level': [16, 16, 18, 12, 14, 16, 13, 15, 17, 16]  # Years of education
}

# Create DataFrame
df = pd.DataFrame(data)

# Standardize the data before PCA (important for PCA performance)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Compute the Correlation Matrix
correlation_matrix = df.corr()

# Apply PCA
pca = PCA(n_components=3)  # We'll extract 3 components
pca_result = pca.fit_transform(scaled_data)

# Create PCA DataFrame
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])

# Plot Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plot PCA Components
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=df['Age'], palette='viridis', s=100)

# Add colorbar using scatter's mappable
plt.title('PCA - First Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Add the colorbar for 'Age'
plt.colorbar(scatter.collections[0], label='Age')

plt.show()

# Explained Variance Ratio (show how much variance each principal component captures)
print("Explained Variance Ratio by PCA Components:")
print(pca.explained_variance_ratio_)

# Plot explained variance to visualize the importance of each principal component
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.7, color='blue')
plt.title('Explained Variance Ratio of PCA Components')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.show()
