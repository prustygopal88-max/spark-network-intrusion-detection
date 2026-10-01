import os
import sys
import json
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.spark.spark_session import setup_spark_environment
setup_spark_environment()

from src.models.predictor import (
    load_inference_artifacts,
    predict_single_flow,
    predict_batch_flows,
    get_feature_columns
)
from src.evaluation.xai_explainer import explain_prediction

app = FastAPI(title="NetGuard AI - Enterprise Network Intrusion Detection", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load metadata and presets on startup
PRESETS_FILE = os.path.join(PROJECT_ROOT, "results", "sample_presets.json")
PREP_REPORT_FILE = os.path.join(PROJECT_ROOT, "results", "preprocessing_report.json")
MODEL_REPORT_FILE = os.path.join(PROJECT_ROOT, "results", "model_comparison.json")
DEMO_CSV_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "demo_batch_sample.csv")

_CACHE = {
    "presets": {},
    "prep_meta": {},
    "models_meta": []
}

def load_cached_data():
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, "r") as f:
            _CACHE["presets"] = json.load(f)
    if os.path.exists(PREP_REPORT_FILE):
        with open(PREP_REPORT_FILE, "r") as f:
            _CACHE["prep_meta"] = json.load(f)
    if os.path.exists(MODEL_REPORT_FILE):
        with open(MODEL_REPORT_FILE, "r") as f:
            _CACHE["models_meta"] = json.load(f)

load_cached_data()

# Ensure Spark models are loaded at server start
@app.on_event("startup")
def startup_event():
    print("[FastAPI] Warming up PySpark MLlib Engine and loading saved models...")
    try:
        load_inference_artifacts()
        print("[FastAPI] PySpark MLlib Engine warmed up and models loaded successfully!")
    except Exception as e:
        print(f"[FastAPI] Error warming up Spark: {e}")

@app.get("/api/metadata")
def get_metadata():
    return JSONResponse(content={
        "preprocessing": _CACHE["prep_meta"],
        "models": _CACHE["models_meta"],
        "feature_count": len(_CACHE["prep_meta"].get("feature_columns", []))
    })

@app.get("/api/presets")
def get_presets():
    return JSONResponse(content=_CACHE["presets"])

@app.post("/api/predict/single")
async def api_predict_single(payload: dict):
    try:
        features = payload.get("features", {})
        model_name = payload.get("model_name", "Random Forest")
        
        # Run inference through PySpark MLlib model
        pred = predict_single_flow(features, model_name=model_name)
        
        # Generate Reason-Based Explainable AI report
        xai = explain_prediction(features, pred)
        
        return JSONResponse(content={
            "prediction": pred,
            "xai": xai
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/batch")
async def api_predict_batch(file: UploadFile = File(None), use_demo: bool = Form(False)):
    try:
        if use_demo:
            if not os.path.exists(DEMO_CSV_FILE):
                raise HTTPException(status_code=404, detail="Demo dataset not found")
            df = pd.read_csv(DEMO_CSV_FILE)
            filename = "demo_batch_sample.csv"
        elif file is not None:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents), low_memory=False)
            filename = file.filename
        else:
            raise HTTPException(status_code=400, detail="No file uploaded and use_demo is False")

        # Run batch prediction through PySpark MLlib
        res_df, summary = predict_batch_flows(df, model_name="Random Forest", max_rows=1000)
        
        # Prepare sample for table preview
        preview_cols = ["Predicted_Class", "Threat_Score (%)", "Confidence (%)", "destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets", "flow_bytes_s"]
        valid_cols = [c for c in preview_cols if c in res_df.columns]
        preview_records = res_df[valid_cols].head(100).to_dict(orient="records")

        # Store latest batch result in temporary CSV for download
        temp_csv_path = os.path.join(PROJECT_ROOT, "results", "last_batch_prediction.csv")
        res_df.to_csv(temp_csv_path, index=False)

        return JSONResponse(content={
            "summary": summary,
            "filename": filename,
            "preview": preview_records
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-batch-results")
def download_batch_results():
    temp_csv_path = os.path.join(PROJECT_ROOT, "results", "last_batch_prediction.csv")
    if not os.path.exists(temp_csv_path):
        raise HTTPException(status_code=404, detail="No batch predictions available for download.")
    
    with open(temp_csv_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    return StreamingResponse(
        io.StringIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=netguard_spark_nids_predictions.csv"}
    )

# High-Tech Cyber Web Dashboard HTML
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>NetGuard AI Server Running. Template not found.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
