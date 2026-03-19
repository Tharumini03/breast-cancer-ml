import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.model_selection import cross_val_score

# ---- Load Preprocessed Data ----
with open('preprocessed_data.pkl', 'rb') as f:
    d = pickle.load(f)

X_train = d['X_train_selected']
X_test  = d['X_test_selected']
y_train = d['y_train']
y_test  = d['y_test']

print("✅ Data loaded successfully!")
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# ---- Define Models ----
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

# ---- Train & Evaluate ----
results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    # Cross-validation (5-fold)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4),
        'CV Mean': round(cv_mean, 4),
        'CV Std': round(cv_std, 4)
    })

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  CV Acc    : {cv_mean:.4f} ± {cv_std:.4f}")

# ---- Summary Table ----
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

print("\n📊 BASELINE MODEL COMPARISON")
print("="*75)
print(results_df.to_string(index=False))
print("="*75)

# Save to CSV for your paper
results_df.to_csv('baseline_results.csv', index=False)
print("\n✅ Results saved to baseline_results.csv")

# ---- Confusion Matrices ----
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    axes[i].set_title(name)
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')

axes[-1].axis('off')  # Hide empty subplot
plt.suptitle('Confusion Matrices - Baseline Models', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
print("✅ Confusion matrices saved!")

# ---- ROC Curves ----
plt.figure(figsize=(9, 6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Baseline Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()
print("✅ ROC curves saved!")

# ---- Bar Chart Comparison ----
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

x = np.arange(len(metrics))
width = 0.15

fig, ax = plt.subplots(figsize=(14, 6))

for i, row in results_df.iterrows():
    values = [row[m] for m in metrics]
    ax.bar(x + i * width, values, width, label=row['Model'])

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Baseline Model Performance Comparison')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics)
ax.set_ylim(0.8, 1.02)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("✅ Comparison chart saved!")