import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

# ---- Load Preprocessed Data ----
with open('preprocessed_data.pkl', 'rb') as f:
    d = pickle.load(f)

X_train = d['X_train_selected']
X_test  = d['X_test_selected']
y_train = d['y_train']
y_test  = d['y_test']

print("✅ Data loaded!")

# ---- Helper: Evaluate any model ----
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    cv     = cross_val_score(model, X_tr, y_tr, cv=5, scoring='accuracy')

    metrics = {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_te, y_pred), 4),
        'Precision': round(precision_score(y_te, y_pred), 4),
        'Recall'   : round(recall_score(y_te, y_pred), 4),
        'F1 Score' : round(f1_score(y_te, y_pred), 4),
        'ROC-AUC'  : round(roc_auc_score(y_te, y_prob), 4),
        'CV Mean'  : round(cv.mean(), 4),
        'CV Std'   : round(cv.std(), 4)
    }

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    for k, v in metrics.items():
        if k != 'Model':
            print(f"  {k:12}: {v}")
    return metrics, model

print("\n" + "="*55)
print("  IMPROVEMENT 1: Hyperparameter Tuning")
print("="*55)

# ---- Tune Logistic Regression ----
lr_params = {
    'C'      : [0.01, 0.1, 1, 10, 100],
    'solver' : ['lbfgs', 'liblinear'],
    'penalty': ['l1', 'l2']
}

# l1 only works with liblinear, so filter valid combos
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    [{'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
     {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs'],     'penalty': ['l2']}],
    cv=5, scoring='f1', n_jobs=-1
)
lr_grid.fit(X_train, y_train)
print(f"\n  Best LR params : {lr_grid.best_params_}")
print(f"  Best CV F1     : {lr_grid.best_score_:.4f}")

# ---- Tune SVM ----
svm_params = [
    {'kernel': ['rbf'],    'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.001]},
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}
]
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_params, cv=5, scoring='f1', n_jobs=-1
)
svm_grid.fit(X_train, y_train)
print(f"\n  Best SVM params: {svm_grid.best_params_}")
print(f"  Best CV F1     : {svm_grid.best_score_:.4f}")

# Evaluate tuned models
tuned_results = []
r, tuned_lr  = evaluate_model("Tuned Logistic Regression", lr_grid.best_estimator_,
                                X_train, X_test, y_train, y_test)
tuned_results.append(r)

r, tuned_svm = evaluate_model("Tuned SVM", svm_grid.best_estimator_,
                               X_train, X_test, y_train, y_test)
tuned_results.append(r)

print("\n" + "="*55)
print("  IMPROVEMENT 2: Ensemble Methods")
print("="*55)

# ---- Random Forest ----
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth'   : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params, cv=5, scoring='f1', n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print(f"\n  Best RF params : {rf_grid.best_params_}")

r, tuned_rf = evaluate_model("Random Forest (Tuned)", rf_grid.best_estimator_,
                              X_train, X_test, y_train, y_test)
tuned_results.append(r)

# ---- Gradient Boosting ----
gb_params = {
    'n_estimators' : [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth'    : [3, 4, 5]
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params, cv=5, scoring='f1', n_jobs=-1
)
gb_grid.fit(X_train, y_train)
print(f"\n  Best GB params : {gb_grid.best_params_}")

r, tuned_gb = evaluate_model("Gradient Boosting (Tuned)", gb_grid.best_estimator_,
                              X_train, X_test, y_train, y_test)
tuned_results.append(r)

print("\n" + "="*55)
print("  IMPROVEMENT 3: Voting Ensemble")
print("="*55)

voting_clf = VotingClassifier(
    estimators=[
        ('lr',  lr_grid.best_estimator_),
        ('svm', svm_grid.best_estimator_),
        ('rf',  rf_grid.best_estimator_)
    ],
    voting='soft'   # uses predicted probabilities instead of hard votes
)

r, tuned_voting = evaluate_model("Soft Voting Ensemble", voting_clf,
                                  X_train, X_test, y_train, y_test)
tuned_results.append(r)

print("\n" + "="*55)
print("  IMPROVEMENT 4: Noise Robustness Test")
print("="*55)

best_model = tuned_svm   # swap with your best model if different
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
noise_results = []

for noise in noise_levels:
    X_noisy = X_test + np.random.normal(0, noise, X_test.shape)
    y_pred  = best_model.predict(X_noisy)
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    noise_results.append({'Noise Level': noise, 'Accuracy': acc, 'F1 Score': f1})
    print(f"  Noise={noise:.2f}  →  Accuracy: {acc:.4f}  F1: {f1:.4f}")

noise_df = pd.DataFrame(noise_results)

# Plot noise robustness
plt.figure(figsize=(9, 5))
plt.plot(noise_df['Noise Level'], noise_df['Accuracy'], marker='o', label='Accuracy')
plt.plot(noise_df['Noise Level'], noise_df['F1 Score'], marker='s', label='F1 Score')
plt.xlabel('Noise Level (std)')
plt.ylabel('Score')
plt.title('Model Robustness Under Gaussian Noise (Tuned SVM)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('noise_robustness.png', dpi=150)
plt.show()
print("✅ Noise robustness plot saved!")

# ---- Final Summary ----
improved_df = pd.DataFrame(tuned_results)
improved_df = improved_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

print("\n📊 IMPROVED MODEL COMPARISON")
print("="*80)
print(improved_df.to_string(index=False))
print("="*80)

improved_df.to_csv('improved_results.csv', index=False)
print("\n✅ Improved results saved to improved_results.csv")

# ---- ROC Curves for all improved models ----
plt.figure(figsize=(9, 6))
for name, model in [("Tuned LR", tuned_lr), ("Tuned SVM", tuned_svm),
                    ("Random Forest", tuned_rf), ("Gradient Boosting", tuned_gb),
                    ("Voting Ensemble", tuned_voting)]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Improved Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('improved_roc_curves.png', dpi=150)
plt.show()
print("✅ Improved ROC curves saved!")