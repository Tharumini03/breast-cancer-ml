import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = Malignant, 1 = Benign

# ---- 1. Basic Info ----
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum().sum())  # Should be 0
print("\nBasic Statistics:\n", df.describe())

# ---- 2. Class Distribution ----
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='Set2')
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.title('Class Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

# ---- 3. Correlation Heatmap ----
plt.figure(figsize=(18, 14))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# ---- 4. Feature Distributions ----
# Pick the most important features to visualize
key_features = ['mean radius', 'mean texture', 'mean area', 
                'mean smoothness', 'mean concavity']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    axes[i].hist(df[df['target'] == 0][feature], 
                 alpha=0.6, label='Malignant', color='red', bins=30)
    axes[i].hist(df[df['target'] == 1][feature], 
                 alpha=0.6, label='Benign', color='green', bins=30)
    axes[i].set_title(feature)
    axes[i].legend()

plt.suptitle('Feature Distributions by Class', fontsize=14)
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

# ---- 5. Boxplots ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, feature in enumerate(['mean radius', 'mean area', 'mean concavity']):
    df.boxplot(column=feature, by='target', ax=axes[i])
    axes[i].set_xticklabels(['Malignant', 'Benign'])
    axes[i].set_title(feature)

plt.suptitle('Boxplots of Key Features by Class')
plt.tight_layout()
plt.savefig('boxplots.png')
plt.show()