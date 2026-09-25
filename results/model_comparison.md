# Model Evaluation & Performance Comparison

## Tested Models on CICIDS2017 Intrusion Detection

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | Train Time (s) | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 99.0% | 98.35% | 99.34% | 98.84% | 0.9975 | 11.94s | 1.4s |
| **Random Forest** | 99.99% | 100.0% | 99.98% | 99.99% | 1.0 | 7.01s | 1.23s |
| **Gradient-Boosted Trees** | 99.96% | 99.95% | 99.96% | 99.95% | 0.9999 | 31.66s | 0.88s |

## Confusion Matrix Breakdown

| Model | True Negatives (BENIGN) | False Positives | False Negatives | True Positives (PortScan) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 24,030 | 304 | 120 | 18,128 |
| **Random Forest** | 24,334 | 0 | 4 | 18,244 |
| **Gradient-Boosted Trees** | 24,324 | 10 | 8 | 18,240 |
