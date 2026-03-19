import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()

# Convert to DataFrame for easier viewing
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = Malignant, 1 = Benign

# Quick look
print(df.shape)        # Should print (569, 31)
print(df.head())       # First 5 rows
print(df['target'].value_counts())  # Class distribution