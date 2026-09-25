import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_predictions(predictions_df, model_name="Model", save_dir="results"):
    """
    Computes comprehensive evaluation metrics from PySpark predictions DataFrame:
    - Accuracy, Precision, Recall, F1-Score
    - ROC-AUC
    - Confusion Matrix (TN, FP, FN, TP)
    - Saves confusion matrix plot to results/figures/
    """
    os.makedirs(save_dir, exist_ok=True)
    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Extract label and prediction vectors from Spark DataFrame
    pdf = predictions_df.select("label", "prediction", "probability").toPandas()
    y_true = pdf["label"].astype(int).values
    y_pred = pdf["prediction"].astype(int).values
    
    # Extract positive class probability if available (vector [p0, p1])
    try:
        y_prob = np.array([float(prob[1]) for prob in pdf["probability"]])
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float(roc_auc_score(y_true, y_pred))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "total_test_samples": len(y_true)
    }

    # Generate Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BENIGN (0)", "PortScan (1)"],
        yticklabels=["BENIGN (0)", "PortScan (1)"]
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, f"{model_name.lower().replace(' ', '_')}_cm.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    metrics["confusion_matrix_plot"] = cm_path

    return metrics
