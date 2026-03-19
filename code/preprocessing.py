import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# ---- Load Data ----
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print("Original dataset shape:", X.shape)  # (569, 30)

# ---- Split BEFORE scaling (very important!) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 80% train, 20% test
    random_state=42,      # reproducibility
    stratify=y            # keeps class ratio balanced in both splits
)

print("Training set size:", X_train.shape)   # ~455 samples
print("Testing set size:", X_test.shape)     # ~114 samples

# Verify class balance is maintained
unique, counts = np.unique(y_train, return_counts=True)
print("\nTraining class distribution:", dict(zip(unique, counts)))
unique, counts = np.unique(y_test, return_counts=True)
print("Testing class distribution:", dict(zip(unique, counts)))

# ---- Scale Features ----
scaler = StandardScaler()

# Fit ONLY on training data, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # Notice: transform only, NOT fit_transform!

print("\nBefore scaling - mean of first feature:", X_train[:, 0].mean().round(2))
print("After scaling  - mean of first feature:", X_train_scaled[:, 0].mean().round(2))  # ~0.0

# ---- Select Best Features ----
selector = SelectKBest(score_func=f_classif, k=15)  # Keep top 15 features

X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# See which features were selected
selected_mask = selector.get_support()
selected_features = feature_names[selected_mask]

print("\nSelected features:")
for i, name in enumerate(selected_features):
    print(f"  {i+1}. {name}")

print("\nShape after feature selection:", X_train_selected.shape)  # (455, 15)

# ---- Package everything neatly for use in models.py ----
import pickle

preprocessing_data = {
    'X_train_scaled': X_train_scaled,
    'X_test_scaled': X_test_scaled,
    'X_train_selected': X_train_selected,
    'X_test_selected': X_test_selected,
    'y_train': y_train,
    'y_test': y_test,
    'selected_features': selected_features,
    'scaler': scaler,
    'selector': selector
}

with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(preprocessing_data, f)

print("\n✅ Preprocessed data saved to preprocessed_data.pkl")