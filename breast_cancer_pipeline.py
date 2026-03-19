import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
 
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)


#   LOAD AND PREPROCESS DATA
data = load_breast_cancer()
X    = data.data
y    = data.target
 
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# Scale
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
 
# Select top 15 features
selector         = SelectKBest(score_func=f_classif, k=15)
X_train_sel      = selector.fit_transform(X_train_scaled, y_train)
X_test_sel       = selector.transform(X_test_scaled)
selected_features = data.feature_names[selector.get_support()]
 
print(f"  Train: {X_train_sel.shape}  |  Test: {X_test_sel.shape}")
print(f"  Selected features: {list(selected_features)}\n")

#   TRAIN AND EVALUATE ALL MODELS
# get all metrics for one model
def get_metrics(name, model, tag):
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:, 1]
    cv     = cross_val_score(model, X_train_sel, y_train, cv=5, scoring='f1')
    return {
        'Model'    : name,
        'Type'     : tag,
        'Accuracy' : round(accuracy_score(y_test, y_pred),  4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall'   : round(recall_score(y_test, y_pred),    4),
        'F1 Score' : round(f1_score(y_test, y_pred),        4),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_prob),   4),
        'CV Mean'  : round(cv.mean(), 4),
        'CV Std'   : round(cv.std(),  4)
    }, model

# Baseline models 
print("Training baseline models...")
baseline_defs = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
    'KNN'                 : KNeighborsClassifier(n_neighbors=5),
    'SVM'                 : SVC(kernel='rbf', probability=True, random_state=42),
    'Decision Tree'       : DecisionTreeClassifier(random_state=42),
    'Naive Bayes'         : GaussianNB()
}
 
baseline_results = []
baseline_models  = {}
for name, model in baseline_defs.items():
    row, fitted = get_metrics(name, model, 'Baseline')
    baseline_results.append(row)
    baseline_models[name] = fitted
    print(f"  {name:25}  F1={row['F1 Score']}  AUC={row['ROC-AUC']}")

# Improved models (GridSearchCV tuning) 
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    [{'C': [0.01,0.1,1,10,100], 'solver':['liblinear'], 'penalty':['l1','l2']},
     {'C': [0.01,0.1,1,10,100], 'solver':['lbfgs'],     'penalty':['l2']}],
    cv=5, scoring='f1'
)
lr_grid.fit(X_train_sel, y_train)
 
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    [{'kernel':['rbf'],    'C':[0.1,1,10,100], 'gamma':['scale','auto',0.01,0.001]},
     {'kernel':['linear'], 'C':[0.1,1,10,100]}],
    cv=5, scoring='f1'
)
svm_grid.fit(X_train_sel, y_train)
 
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators':[100,200,300], 'max_depth':[None,5,10,15], 'min_samples_split':[2,5,10]},
    cv=5, scoring='f1'
)
rf_grid.fit(X_train_sel, y_train)

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    {'n_estimators':[100,200], 'learning_rate':[0.01,0.05,0.1], 'max_depth':[3,4,5]},
    cv=5, scoring='f1'
)
gb_grid.fit(X_train_sel, y_train)
 
voting_clf = VotingClassifier(
    estimators=[('lr', lr_grid.best_estimator_),
                ('svm', svm_grid.best_estimator_),
                ('rf', rf_grid.best_estimator_)],
    voting='soft'
)

improved_defs = {
    'Tuned LR'          : lr_grid.best_estimator_,
    'Tuned SVM'         : svm_grid.best_estimator_,
    'Random Forest'     : rf_grid.best_estimator_,
    'Gradient Boosting' : gb_grid.best_estimator_,
    'Voting Ensemble'   : voting_clf
}
 
improved_results = []
improved_models  = {}
for name, model in improved_defs.items():
    row, fitted = get_metrics(name, model, 'Improved')
    improved_results.append(row)
    improved_models[name] = fitted
    print(f"  {name:25}  F1={row['F1 Score']}  AUC={row['ROC-AUC']}")

#  Save results table 
all_df = pd.DataFrame(baseline_results + improved_results)
all_df.to_csv('final_results.csv', index=False)
print(all_df[['Model','Accuracy','Precision','Recall','F1 Score','ROC-AUC']].to_string(index=False))
 
#  GENERATE FIGURES
#  ROC Curves (Baseline vs Improved) 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
for name, model in baseline_models.items():
    y_prob = model.predict_proba(X_test_sel)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f'{name} ({roc_auc_score(y_test, y_prob):.3f})')
axes[0].plot([0,1],[0,1],'k--')
axes[0].set_title('ROC Curves - Baseline Models')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)
 
for name, model in improved_models.items():
    y_prob = model.predict_proba(X_test_sel)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, label=f'{name} ({roc_auc_score(y_test, y_prob):.3f})')
axes[1].plot([0,1],[0,1],'k--')
axes[1].set_title('ROC Curves - Improved Models')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3)
 
plt.suptitle('ROC Curve Comparison')
plt.tight_layout()
plt.savefig('final_roc_curves.png', dpi=150)
plt.show()

# Confusion Matrix (Best Model = Tuned SVM) 
best_model  = improved_models['Tuned SVM']
y_pred_best = best_model.predict(X_test_sel)
cm          = confusion_matrix(y_test, y_pred_best)
 
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.title('Confusion Matrix - Tuned SVM')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png', dpi=150)
plt.show()
 
print("\nClassification Report - Tuned SVM")
print(classification_report(y_test, y_pred_best, target_names=['Malignant','Benign']))
 
#  Feature Importance (Random Forest) 
importances = improved_models['Random Forest'].feature_importances_
indices     = np.argsort(importances)[::-1]
 
plt.figure(figsize=(10, 4))
plt.bar(range(len(selected_features)), importances[indices], color='steelblue')
plt.xticks(range(len(selected_features)),
           [selected_features[i] for i in indices],
           rotation=45, ha='right', fontsize=8)
plt.title('Feature Importance - Random Forest')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

#  Cross-Validation Box Plot 
all_models = {**baseline_models, **improved_models}
cv_data    = {}
for name, model in all_models.items():
    cv_data[name] = cross_val_score(model, X_train_sel, y_train, cv=5, scoring='f1')
 
cv_df = pd.DataFrame(cv_data)
plt.figure(figsize=(12, 5))
cv_df.boxplot(rot=30)
plt.title('5-Fold Cross Validation F1 Scores - All Models')
plt.ylabel('F1 Score')
plt.ylim(0.88, 1.02)
plt.tight_layout()
plt.savefig('cv_boxplot.png', dpi=150)
plt.show()

#  Noise Robustness (Tuned SVM) 
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
noise_acc    = []
noise_f1     = []
 
for sigma in noise_levels:
    X_noisy = X_test_sel + np.random.normal(0, sigma, X_test_sel.shape)
    y_pred  = best_model.predict(X_noisy)
    noise_acc.append(accuracy_score(y_test, y_pred))
    noise_f1.append(f1_score(y_test, y_pred))
 
plt.figure(figsize=(7, 4))
plt.plot(noise_levels, noise_acc, marker='o', label='Accuracy')
plt.plot(noise_levels, noise_f1,  marker='s', label='F1 Score')
plt.xlabel('Noise Level (sigma)')
plt.ylabel('Score')
plt.title('Noise Robustness - Tuned SVM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('noise_robustness.png', dpi=150)
plt.show()

#   FINAL SUMMARY
print("\n" + "="*50)
print("  SUMMARY")
print("="*50)
 
best_row  = all_df.sort_values('F1 Score', ascending=False).iloc[0]
base_best = all_df[all_df['Type']=='Baseline'].sort_values('F1 Score', ascending=False).iloc[0]
 
print(f"Best Baseline : {base_best['Model']}  F1={base_best['F1 Score']}  AUC={base_best['ROC-AUC']}")
print(f"Best Improved : {best_row['Model']}  F1={best_row['F1 Score']}  AUC={best_row['ROC-AUC']}")
print(f"F1 Gain       : +{round(best_row['F1 Score'] - base_best['F1 Score'], 4)}")