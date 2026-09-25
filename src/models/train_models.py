import os
import sys
import time
import json
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark.spark_session import get_spark_session, stop_spark_session
from src.evaluation.metrics import evaluate_predictions
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)

def train_and_evaluate_all():
    print("================ STAGE 4: MODEL TRAINING & EVALUATION ================\n")
    start_total = time.time()

    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    train_path = os.path.join(processed_dir, "train.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    models_dir = os.path.join(results_dir, "models")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Start SparkSession
    spark = get_spark_session(app_name="CICIDS2017-ModelTraining")
    print(f"SparkSession initialized (v{spark.version})")

    try:
        # 2. Load preprocessed train and test datasets
        print("Loading preprocessed train and test partitions from Parquet...")
        train_df = spark.read.parquet(train_path)
        test_df = spark.read.parquet(test_path)
        train_df.cache()
        test_df.cache()

        train_count = train_df.count()
        test_count = test_df.count()
        print(f"Train records: {train_count:,} | Test records: {test_count:,}\n")

        all_results = []

        # ==========================================
        # MODEL 1: LOGISTIC REGRESSION
        # ==========================================
        print(">>> [1/3] Training Logistic Regression...")
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=50,
            regParam=0.01,
            elasticNetParam=0.0
        )
        
        t0 = time.time()
        lr_model = lr.fit(train_df)
        lr_train_time = round(time.time() - t0, 2)
        print(f"    Trained in {lr_train_time} seconds.")

        # Save model
        lr_save_path = os.path.join(models_dir, "logistic_regression_model")
        lr_model.write().overwrite().save(lr_save_path)
        print(f"    Model saved to: {lr_save_path}")

        # Predict on test set
        t0 = time.time()
        lr_predictions = lr_model.transform(test_df)
        lr_predictions.cache()
        _ = lr_predictions.count()  # materialize
        lr_pred_time = round(time.time() - t0, 2)
        print(f"    Test prediction completed in {lr_pred_time} seconds.")

        # Evaluate
        lr_metrics = evaluate_predictions(lr_predictions, model_name="Logistic Regression", save_dir=results_dir)
        lr_metrics["train_time_sec"] = lr_train_time
        lr_metrics["inference_time_sec"] = lr_pred_time
        all_results.append(lr_metrics)
        print(f"    Accuracy:  {lr_metrics['accuracy'] * 100:.2f}%")
        print(f"    Precision: {lr_metrics['precision'] * 100:.2f}%")
        print(f"    Recall:    {lr_metrics['recall'] * 100:.2f}%")
        print(f"    F1-Score:  {lr_metrics['f1_score'] * 100:.2f}%")
        print(f"    ROC-AUC:   {lr_metrics['roc_auc']:.4f}\n")

        # ==========================================
        # MODEL 2: RANDOM FOREST CLASSIFIER
        # ==========================================
        print(">>> [2/3] Training Random Forest Classifier...")
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=30,
            maxDepth=10,
            seed=42
        )
        
        t0 = time.time()
        rf_model = rf.fit(train_df)
        rf_train_time = round(time.time() - t0, 2)
        print(f"    Trained in {rf_train_time} seconds.")

        # Save model
        rf_save_path = os.path.join(models_dir, "random_forest_model")
        rf_model.write().overwrite().save(rf_save_path)
        print(f"    Model saved to: {rf_save_path}")

        # Predict on test set
        t0 = time.time()
        rf_predictions = rf_model.transform(test_df)
        rf_predictions.cache()
        _ = rf_predictions.count()  # materialize
        rf_pred_time = round(time.time() - t0, 2)
        print(f"    Test prediction completed in {rf_pred_time} seconds.")

        # Evaluate
        rf_metrics = evaluate_predictions(rf_predictions, model_name="Random Forest", save_dir=results_dir)
        rf_metrics["train_time_sec"] = rf_train_time
        rf_metrics["inference_time_sec"] = rf_pred_time
        all_results.append(rf_metrics)
        print(f"    Accuracy:  {rf_metrics['accuracy'] * 100:.2f}%")
        print(f"    Precision: {rf_metrics['precision'] * 100:.2f}%")
        print(f"    Recall:    {rf_metrics['recall'] * 100:.2f}%")
        print(f"    F1-Score:  {rf_metrics['f1_score'] * 100:.2f}%")
        print(f"    ROC-AUC:   {rf_metrics['roc_auc']:.4f}\n")

        # ==========================================
        # MODEL 3: GRADIENT-BOOSTED TREES (GBT)
        # ==========================================
        print(">>> [3/3] Training Gradient-Boosted Trees (GBT)...")
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=30,
            maxDepth=5,
            seed=42
        )
        
        t0 = time.time()
        gbt_model = gbt.fit(train_df)
        gbt_train_time = round(time.time() - t0, 2)
        print(f"    Trained in {gbt_train_time} seconds.")

        # Save model
        gbt_save_path = os.path.join(models_dir, "gbt_model")
        gbt_model.write().overwrite().save(gbt_save_path)
        print(f"    Model saved to: {gbt_save_path}")

        # Predict on test set
        t0 = time.time()
        gbt_predictions = gbt_model.transform(test_df)
        gbt_predictions.cache()
        _ = gbt_predictions.count()  # materialize
        gbt_pred_time = round(time.time() - t0, 2)
        print(f"    Test prediction completed in {gbt_pred_time} seconds.")

        # Evaluate
        gbt_metrics = evaluate_predictions(gbt_predictions, model_name="Gradient-Boosted Trees", save_dir=results_dir)
        gbt_metrics["train_time_sec"] = gbt_train_time
        gbt_metrics["inference_time_sec"] = gbt_pred_time
        all_results.append(gbt_metrics)
        print(f"    Accuracy:  {gbt_metrics['accuracy'] * 100:.2f}%")
        print(f"    Precision: {gbt_metrics['precision'] * 100:.2f}%")
        print(f"    Recall:    {gbt_metrics['recall'] * 100:.2f}%")
        print(f"    F1-Score:  {gbt_metrics['f1_score'] * 100:.2f}%")
        print(f"    ROC-AUC:   {gbt_metrics['roc_auc']:.4f}\n")

        # ==========================================
        # COMPARISON & SUMMARY REPORT
        # ==========================================
        print(">>> Generating Model Comparison Summary...")
        summary_rows = []
        for r in all_results:
            cm = r["confusion_matrix"]
            summary_rows.append({
                "Model": r["model_name"],
                "Accuracy (%)": round(r["accuracy"] * 100, 2),
                "Precision (%)": round(r["precision"] * 100, 2),
                "Recall (%)": round(r["recall"] * 100, 2),
                "F1-Score (%)": round(r["f1_score"] * 100, 2),
                "ROC-AUC": r["roc_auc"],
                "Train Time (s)": r["train_time_sec"],
                "Inference Time (s)": r["inference_time_sec"],
                "True Negatives": cm["tn"],
                "False Positives": cm["fp"],
                "False Negatives": cm["fn"],
                "True Positives": cm["tp"]
            })

        summary_df = pd.DataFrame(summary_rows)
        print("\n" + summary_df.to_string(index=False) + "\n")

        # Save comparison outputs
        json_summary_path = os.path.join(results_dir, "model_comparison.json")
        csv_summary_path = os.path.join(results_dir, "model_comparison.csv")
        md_summary_path = os.path.join(results_dir, "model_comparison.md")

        with open(json_summary_path, "w") as jf:
            json.dump(all_results, jf, indent=4)
        summary_df.to_csv(csv_summary_path, index=False)

        with open(md_summary_path, "w") as mf:
            mf.write(f"""# Model Evaluation & Performance Comparison

## Tested Models on CICIDS2017 Intrusion Detection

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | Train Time (s) | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
""")
            for _, row in summary_df.iterrows():
                mf.write(f"| **{row['Model']}** | {row['Accuracy (%)']}% | {row['Precision (%)']}% | {row['Recall (%)']}% | {row['F1-Score (%)']}% | {row['ROC-AUC']} | {row['Train Time (s)']}s | {row['Inference Time (s)']}s |\n")

            mf.write("\n## Confusion Matrix Breakdown\n\n")
            mf.write("| Model | True Negatives (BENIGN) | False Positives | False Negatives | True Positives (PortScan) |\n")
            mf.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for _, row in summary_df.iterrows():
                mf.write(f"| **{row['Model']}** | {row['True Negatives']:,} | {row['False Positives']:,} | {row['False Negatives']:,} | {row['True Positives']:,} |\n")

        # Generate comparison bar chart
        plt.figure(figsize=(10, 6))
        plot_df = summary_df.set_index("Model")[["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]]
        plot_df.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
        plt.title("Comparative Performance across PySpark MLlib Models", fontsize=14, fontweight="bold")
        plt.ylabel("Score (%)", fontsize=12)
        plt.ylim(90, 101)
        plt.xticks(rotation=0, fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(loc="lower right")
        plt.tight_layout()
        chart_path = os.path.join(figures_dir, "model_comparison_bar.png")
        plt.savefig(chart_path, dpi=200)
        plt.close()
        print(f"Comparison chart saved to: {chart_path}")

        total_duration = round(time.time() - start_total, 2)
        print(f"\n================ STAGE 4 COMPLETE ({total_duration}s) ================")

    finally:
        stop_spark_session(spark)

if __name__ == "__main__":
    train_and_evaluate_all()
