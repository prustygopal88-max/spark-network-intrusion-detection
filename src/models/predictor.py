import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark.spark_session import get_spark_session
from src.preprocessing.cleaner import clean_column_name
from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    LogisticRegressionModel,
    RandomForestClassificationModel,
    GBTClassificationModel
)

_SPARK = None
_PIPELINE = None
_MODELS = {}
_FEATURE_COLS = None

def get_spark():
    global _SPARK
    if _SPARK is None:
        _SPARK = get_spark_session(app_name="StreamlitNIDSInference")
    return _SPARK

def get_feature_columns():
    global _FEATURE_COLS
    if _FEATURE_COLS is None:
        report_path = os.path.join(PROJECT_ROOT, "results", "preprocessing_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                meta = json.load(f)
                _FEATURE_COLS = meta.get("feature_columns", [])
        else:
            raise FileNotFoundError("Preprocessing report not found. Run Stage 3 first.")
    return _FEATURE_COLS

def load_inference_artifacts():
    """Loads and caches SparkSession, Feature PipelineModel, and MLlib models."""
    global _PIPELINE, _MODELS
    spark = get_spark()

    models_dir = os.path.join(PROJECT_ROOT, "results", "models")
    pipeline_path = os.path.join(models_dir, "feature_pipeline_model")
    rf_path = os.path.join(models_dir, "random_forest_model")
    gbt_path = os.path.join(models_dir, "gbt_model")
    lr_path = os.path.join(models_dir, "logistic_regression_model")

    if _PIPELINE is None:
        _PIPELINE = PipelineModel.load(pipeline_path)

    if not _MODELS:
        _MODELS["Random Forest"] = RandomForestClassificationModel.load(rf_path)
        _MODELS["Gradient-Boosted Trees"] = GBTClassificationModel.load(gbt_path)
        _MODELS["Logistic Regression"] = LogisticRegressionModel.load(lr_path)

    return spark, _PIPELINE, _MODELS

def predict_single_flow(feature_dict, model_name="Random Forest"):
    """
    Runs single-flow network intrusion inference:
    Returns dict:
      - prediction_label: 'BENIGN' or 'PORTSCAN'
      - confidence: float (0.0 to 1.0)
      - benign_prob: float
      - portscan_prob: float
      - model_used: str
    """
    spark, pipeline, models = load_inference_artifacts()
    feature_cols = get_feature_columns()

    # Build clean single-row dataframe with all 68 required features
    row_data = {}
    for col in feature_cols:
        val = feature_dict.get(col, 0.0)
        try:
            row_data[col] = float(val)
        except (ValueError, TypeError):
            row_data[col] = 0.0

    pdf = pd.DataFrame([row_data])
    sdf = spark.createDataFrame(pdf)

    # Transform through VectorAssembler + StandardScaler
    transformed_sdf = pipeline.transform(sdf)

    # Predict with chosen model
    model = models.get(model_name, models["Random Forest"])
    pred_sdf = model.transform(transformed_sdf)

    pred_row = pred_sdf.select("prediction", "probability").first()
    pred_val = int(pred_row["prediction"])
    prob_vec = pred_row["probability"]
    p_benign = float(prob_vec[0])
    p_portscan = float(prob_vec[1])

    label_str = "PORTSCAN" if pred_val == 1 else "BENIGN"
    confidence = p_portscan if pred_val == 1 else p_benign

    return {
        "prediction_label": label_str,
        "class_id": pred_val,
        "confidence": confidence,
        "benign_prob": p_benign,
        "portscan_prob": p_portscan,
        "model_used": model_name
    }

def predict_batch_flows(input_df, model_name="Random Forest", max_rows=10000):
    """
    Runs batch inference on an uploaded pandas DataFrame:
    1. Normalizes column names (strips whitespace, replaces slashes with underscores)
    2. Validates against expected 68 feature columns
    3. Transforms using Spark MLlib Pipeline
    4. Evaluates predictions and probabilities
    5. Returns enriched pandas DataFrame with predictions and summary dict
    """
    spark, pipeline, models = load_inference_artifacts()
    feature_cols = get_feature_columns()

    # Step 1: Clean column names of input
    cleaned_input = input_df.copy()
    col_map = {c: clean_column_name(c) for c in cleaned_input.columns}
    cleaned_input = cleaned_input.rename(columns=col_map)

    # Check for missing features
    missing_features = [col for col in feature_cols if col not in cleaned_input.columns]
    for col in missing_features:
        cleaned_input[col] = 0.0

    # Ensure numeric types
    for col in feature_cols:
        cleaned_input[col] = pd.to_numeric(cleaned_input[col], errors="coerce").fillna(0.0).astype(np.float64)

    # Limit to max_rows for responsive web execution
    if len(cleaned_input) > max_rows:
        cleaned_input = cleaned_input.iloc[:max_rows]

    # Convert to Spark
    sub_df = cleaned_input[feature_cols]
    sdf = spark.createDataFrame(sub_df)

    # Transform with Pipeline
    transformed_sdf = pipeline.transform(sdf)

    # Predict with Model
    model = models.get(model_name, models["Random Forest"])
    pred_sdf = model.transform(transformed_sdf)

    # Pull predictions back to pandas
    results_pdf = pred_sdf.select("prediction", "probability").toPandas()
    
    preds = results_pdf["prediction"].astype(int).values
    probs = [float(p[1]) for p in results_pdf["probability"]]
    confidences = [float(p[1]) if pred == 1 else float(p[0]) for pred, p in zip(preds, results_pdf["probability"])]

    output_df = cleaned_input.copy()
    output_df["Predicted_Class"] = ["PORTSCAN" if p == 1 else "BENIGN" for p in preds]
    output_df["Threat_Score (%)"] = [round(prob * 100, 2) for prob in probs]
    output_df["Confidence (%)"] = [round(conf * 100, 2) for conf in confidences]

    total_count = len(output_df)
    portscan_count = int(np.sum(preds == 1))
    benign_count = int(np.sum(preds == 0))

    summary = {
        "total_analyzed": total_count,
        "benign_count": benign_count,
        "portscan_count": portscan_count,
        "benign_pct": round((benign_count / total_count) * 100, 2) if total_count > 0 else 0,
        "portscan_pct": round((portscan_count / total_count) * 100, 2) if total_count > 0 else 0,
        "missing_features_imputed": len(missing_features),
        "model_used": model_name
    }

    return output_df, summary
