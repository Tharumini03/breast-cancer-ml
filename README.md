# Breast Cancer Diagnosis Using Classical Machine Learning

A research project for **CS3111 - Introduction to Machine Learning**.

This project applies classical machine learning methods to the UCI Wisconsin Breast Cancer Diagnostic (WDBC) dataset to classify tumors as **malignant or benign**. The work is written up as an IEEE-format conference paper.

---

## Results Summary

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression *(baseline)* | 0.9649 | 0.9722 | 0.9914 |
| KNN *(baseline)* | 0.9561 | 0.9650 | 0.9803 |
| SVM *(baseline)* | 0.9474 | 0.9583 | 0.9911 |
| Decision Tree *(baseline)* | 0.9211 | 0.9362 | 0.9226 |
| Naive Bayes *(baseline)* | 0.9298 | 0.9444 | 0.9861 |
| Tuned LR | 0.9561 | 0.9655 | 0.9917 |
| **Tuned SVM** *(best F1)* | **0.9737** | **0.9790** | **0.9931** |
| Random Forest | 0.9561 | 0.9655 | 0.9906 |
| Gradient Boosting | 0.9474 | 0.9589 | 0.9894 |
| Voting Ensemble *(best AUC)* | 0.9561 | 0.9655 | **0.9970** |

---

## Project Structure

```
breast-cancer-ml/
│
├── breast_cancer_pipeline.py   ← main pipeline (run this)
├── final_results.csv           ← all model metrics (generated)
├── requirements.txt            ← dependencies
├── README.md                   ← this file
│
├── figures/                    ← all paper figures (generated)
│   ├── final_roc_curves.png
│   ├── best_model_confusion_matrix.png
│   ├── feature_importance.png
│   ├── cv_boxplot.png
│   └── noise_robustness.png
│
└── paper/
    ├── main.tex                ← IEEE LaTeX paper
    └── IEEEtran.cls            ← IEEE template class file
```

---

## Dataset

**UCI Wisconsin Breast Cancer Diagnostic (WDBC)**
- 569 samples, 30 features, 2 classes (Malignant / Benign)
- Built into `scikit-learn` — no manual download needed
- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

## Pipeline Overview

| Step | Description |
|---|---|
| Preprocessing | Stratified 80/20 split, StandardScaler, SelectKBest (top 15 features) |
| Baseline Models | Logistic Regression, KNN, SVM, Decision Tree, Naive Bayes |
| Improvements | GridSearchCV tuning, Random Forest, Gradient Boosting, Voting Ensemble |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, 5-Fold CV |
| Robustness | Gaussian noise test on best model (Tuned SVM) |

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/breast-cancer-ml.git
cd breast-cancer-ml
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the pipeline**
```bash
python breast_cancer_pipeline.py
```

> ⏱️ Runtime is approximately **3–5 minutes** due to GridSearchCV hyperparameter tuning.

---

## Output Files

After running, these files are generated automatically:

| File | Description | Used in Paper |
|---|---|---|
| `final_results.csv` | All model metrics | Tables I & II |
| `final_roc_curves.png` | ROC curves (baseline vs improved) | Fig. 1 |
| `best_model_confusion_matrix.png` | Confusion matrix — Tuned SVM | Fig. 2 |
| `feature_importance.png` | Random Forest feature importances | Fig. 3 |
| `cv_boxplot.png` | 5-fold CV F1 distributions | Fig. 4 |
| `noise_robustness.png` | Noise robustness of Tuned SVM | Fig. 5 |

---

## Key Findings

- **Tuned SVM** achieved the best F1 score of **0.9790** and correctly identified **41/43 malignant cases** in the test set
- **Voting Ensemble** achieved the highest ROC-AUC of **0.9970**, combining Tuned LR, Tuned SVM, and Random Forest
- **Worst perimeter** was the most important feature (importance score: 0.181), consistent with clinical knowledge that tumor size is a key malignancy indicator
- The model maintained **94.74% accuracy** under Gaussian noise at σ=0.10, demonstrating reasonable clinical robustness

---

## Technologies Used

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

---

## Paper

This project is written up as an **IEEE conference paper** using the official IEEE LaTeX template. The paper covers:

1. Introduction & Background
2. Related Work
3. Methodology
4. Results
5. Discussion
6. Conclusion

---

## Author

** N.G.T.S.GAMAGE **
Department of Computer Science, University of Moratuwa

---

## License

This project is for academic purposes as part of CS3111 - Introduction to Machine Learning.
