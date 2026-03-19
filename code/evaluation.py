import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance

# ---- Load Data ----
with open('preprocessed_data.pkl', 'rb') as f:
    d = pickle.load(f)

X_train          = d['X_train_selected']
X_test           = d['X_test_selected']
y_train          = d['y_train']
y_test           = d['y_test']
selected_features = d['selected_features']

print("✅ Data loaded successfully!")

# ---- Rebuild Best Models (from Phase 4 & 5 results) ----

# Baseline models
baseline_models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'KNN'                 : KNeighborsClassifier(n_neighbors=5),
    'SVM'                 : SVC(kernel='rbf', probability=True, random_state=42),
    'Decision Tree'       : DecisionTreeClassifier(random_state=42),
    'Naive Bayes'         : GaussianNB()
}

# Improved models — replace params with YOUR best params from Phase 5
improved_models = {
    'Tuned LR'            : LogisticRegression(C=10, solver='liblinear',
                                               penalty='l2', max_iter=1000,
                                               random_state=42),
    'Tuned SVM'           : SVC(kernel='rbf', C=10, gamma='scale',
                                probability=True, random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   min_samples_split=2,
                                                   random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=200,
                                                       learning_rate=0.05,
                                                       max_depth=3,
                                                       random_state=42),
}

# Add Voting Ensemble
improved_models['Voting Ensemble'] = VotingClassifier(
    estimators=[
        ('lr',  improved_models['Tuned LR']),
        ('svm', improved_models['Tuned SVM']),
        ('rf',  improved_models['Random Forest'])
    ],
    voting='soft'
)

# ---- Train All ----
print("\nTraining all models...")
for name, model in {**baseline_models, **improved_models}.items():
    model.fit(X_train, y_train)
    print(f"  ✅ {name}")

# ---- Collect Full Metrics ----
def get_metrics(name, model, tag='Baseline'):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv     = cross_val_score(model, X_train, y_train,
                             cv=StratifiedKFold(n_splits=5), scoring='f1')
    return {
        'Model'    : name,
        'Type'     : tag,
        'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall'   : round(recall_score(y_test, y_pred), 4),
        'F1 Score' : round(f1_score(y_test, y_pred), 4),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_prob), 4),
        'CV F1'    : round(cv.mean(), 4),
        'CV Std'   : round(cv.std(), 4)
    }

all_results = []

for name, model in baseline_models.items():
    all_results.append(get_metrics(name, model, 'Baseline'))

for name, model in improved_models.items():
    all_results.append(get_metrics(name, model, 'Improved'))

results_df = pd.DataFrame(all_results)
results_df.to_csv('final_results.csv', index=False)

print("\n📊 COMPLETE RESULTS TABLE")
print("="*85)
print(results_df.to_string(index=False))
print("="*85)

# ---- Baseline vs Improved Bar Chart ----
metrics   = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
baseline  = results_df[results_df['Type'] == 'Baseline']
improved  = results_df[results_df['Type'] == 'Improved']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Baseline
baseline_vals = baseline[metrics].values
x = np.arange(len(metrics))
width = 0.15

for i, row in baseline.iterrows():
    vals = [row[m] for m in metrics]
    axes[0].bar(x + list(baseline['Model']).index(row['Model']) * width,
                vals, width, label=row['Model'])

axes[0].set_title('Baseline Models', fontsize=13)
axes[0].set_xticks(x + width * 2)
axes[0].set_xticklabels(metrics, rotation=15)
axes[0].set_ylim(0.80, 1.02)
axes[0].legend(fontsize=7)
axes[0].set_ylabel('Score')
axes[0].grid(axis='y', alpha=0.3)

# Improved
for i, row in improved.iterrows():
    vals = [row[m] for m in metrics]
    axes[1].bar(x + list(improved['Model']).index(row['Model']) * width,
                vals, width, label=row['Model'])

axes[1].set_title('Improved Models', fontsize=13)
axes[1].set_xticks(x + width * 2)
axes[1].set_xticklabels(metrics, rotation=15)
axes[1].set_ylim(0.80, 1.02)
axes[1].legend(fontsize=7)
axes[1].set_ylabel('Score')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Baseline vs Improved Model Comparison', fontsize=15)
plt.tight_layout()
plt.savefig('baseline_vs_improved.png', dpi=150)
plt.show()
print("✅ Comparison chart saved!")

# ---- Best Model Confusion Matrix ----
best_model      = improved_models['Voting Ensemble']
best_model_name = 'Voting Ensemble'

y_pred = best_model.predict(X_test)

fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Malignant', 'Benign'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=13)

# Annotate TP, TN, FP, FN
ax.text(0, 0.35, 'True\nNegative',  ha='center', color='gray', fontsize=9)
ax.text(1, 0.35, 'False\nPositive', ha='center', color='gray', fontsize=9)
ax.text(0, 1.35, 'False\nNegative', ha='center', color='gray', fontsize=9)
ax.text(1, 1.35, 'True\nPositive',  ha='center', color='gray', fontsize=9)

plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png', dpi=150)
plt.show()
print("✅ Best model confusion matrix saved!")

# Print full classification report
print(f"\n📋 Classification Report — {best_model_name}")
print("="*55)
print(classification_report(y_test, y_pred,
                             target_names=['Malignant', 'Benign']))

# ---- Combined ROC — Baseline vs Improved ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Baseline ROC
for name, model in baseline_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f'{name} ({auc:.3f})')

axes[0].plot([0,1],[0,1],'k--')
axes[0].set_title('ROC Curves — Baseline Models')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=8, loc='lower right')
axes[0].grid(alpha=0.3)

# Improved ROC
for name, model in improved_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[1].plot(fpr, tpr, label=f'{name} ({auc:.3f})')

axes[1].plot([0,1],[0,1],'k--')
axes[1].set_title('ROC Curves — Improved Models')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=8, loc='lower right')
axes[1].grid(alpha=0.3)

plt.suptitle('ROC Curve Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('final_roc_curves.png', dpi=150)
plt.show()
print("✅ Final ROC curves saved!")

# ---- Feature Importance (Random Forest) ----
rf_model    = improved_models['Random Forest']
importances = rf_model.feature_importances_
indices     = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 5))
plt.bar(range(len(selected_features)),
        importances[indices], color='steelblue', alpha=0.8)
plt.xticks(range(len(selected_features)),
           [selected_features[i] for i in indices],
           rotation=45, ha='right', fontsize=9)
plt.title('Feature Importance — Random Forest')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("✅ Feature importance plot saved!")

# Top 5 features
print("\n🔝 Top 5 Most Important Features:")
for i in range(5):
    print(f"  {i+1}. {selected_features[indices[i]]:30} {importances[indices[i]]:.4f}")

# ---- CV Box Plot ----
all_models = {**baseline_models, **improved_models}
cv_data    = {}

for name, model in all_models.items():
    scores = cross_val_score(model, X_train, y_train,
                             cv=StratifiedKFold(n_splits=5), scoring='f1')
    cv_data[name] = scores

cv_df = pd.DataFrame(cv_data)

plt.figure(figsize=(14, 6))
cv_df.boxplot(rot=30)
plt.title('5-Fold Cross-Validation F1 Scores — All Models')
plt.ylabel('F1 Score')
plt.ylim(0.88, 1.02)
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('cv_boxplot.png', dpi=150)
plt.show()
print("✅ CV box plot saved!")

# ---- Final Summary ----
print("\n" + "🏆 "*20)
print("         FINAL EVALUATION SUMMARY")
print("🏆 "*20)

best_row = results_df.sort_values('F1 Score', ascending=False).iloc[0]
base_row = results_df[results_df['Type']=='Baseline'].sort_values(
                                'F1 Score', ascending=False).iloc[0]

print(f"\n  Best Baseline : {base_row['Model']:30} F1={base_row['F1 Score']:.4f}")
print(f"  Best Improved : {best_row['Model']:30} F1={best_row['F1 Score']:.4f}")
print(f"\n  Improvement   : +{(best_row['F1 Score'] - base_row['F1 Score']):.4f} F1 Score")
print(f"  Best Accuracy : {best_row['Accuracy']:.4f}")
print(f"  Best ROC-AUC  : {best_row['ROC-AUC']:.4f}")
print(f"  CV F1 Score   : {best_row['CV F1']:.4f} ± {best_row['CV Std']:.4f}")
print("\n✅ All evaluation files saved and ready for paper writing!")