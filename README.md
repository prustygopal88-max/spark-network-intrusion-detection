# Scalable Network Intrusion Detection and Anomaly Analysis Using Apache Spark

An end-to-end Big Data cybersecurity mini-project that detects network intrusions and scanning anomalies at scale using **Apache Spark (PySpark MLlib)**, **Java OpenJDK 17 LTS**, and an interactive **Streamlit** dashboard.

---

## 📌 Project Overview & Objectives

### Problem Statement
Enterprise networks generate millions of packet flow transactions every second. Traditional, single-node intrusion detection systems frequently suffer from CPU starvation, high memory overhead, and unacceptably high false alarm rates under heavy network load.

### Project Objective
To build an automated, distributed machine learning intrusion detection pipeline on **Apache Spark** that:
1. Ingests and sanitizes large-scale network telemetry.
2. Extracts and standardizes 68 predictive network flow features.
3. Evaluates three distributed classifiers: **Logistic Regression**, **Random Forest**, and **Gradient-Boosted Trees (GBT)**.
4. Provides a presentation-ready Streamlit web interface for single-flow testing and batch classification.

---

## 📊 Dataset Description: CICIDS2017

- **Source:** Canadian Institute for Cybersecurity (CIC) / University of New Brunswick (UNB).
- **Target Partition:** `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- **Volume:** 286,467 raw flow records (73.34 MB).
- **Cleaned Data:** 213,777 validated records, 68 predictive features.
- **Classification Goal:** Binary classification:
  - `BENIGN` (`0`): Normal enterprise background traffic.
  - `PortScan` (`1`): Malicious network reconnaissance and port scanning probes.

---

## 🛠️ Technologies Used

| Layer | Component | Version | Role |
| :--- | :--- | :---: | :--- |
| **Distributed Engine** | **Apache Spark** | 3.5.9 | In-memory distributed computation |
| **Java Runtime** | **Microsoft OpenJDK** | 17.0.20 LTS | JVM execution runtime |
| **Machine Learning** | **PySpark MLlib** | 3.5.9 | VectorAssembler, StandardScaler, RF, GBT, LR |
| **Programming Language** | **Python** | 3.12.10 | Core scripting and pipeline orchestration |
| **Web Dashboard** | **Streamlit** | 1.44.1 | Interactive user interface |
| **Data & Metrics** | **Pandas, NumPy, Scikit-learn** | 2.2, 2.2, 1.8 | Metric evaluation & tabular manipulations |
| **Visualization** | **Matplotlib, Seaborn** | 3.10, 0.13 | Confusion matrix & comparison charts |

---

## 🏛️ System Architecture

```text
[ Raw CICIDS2017 CSV: 286,467 rows ]
                 │
                 ▼
[ Data Cleaning & Normalization (cleaner.py) ]
 ├── Column header standardization (snake_case)
 ├── Label encoding: BENIGN -> 0, PortScan -> 1
 ├── Duplicate removal (72,353 duplicates filtered)
 ├── Infinite & null value sanitization (337 rows dropped)
 └── Zero-variance feature pruning (10 constant columns removed)
                 │
                 ▼
[ PySpark Distributed Feature Pipeline (feature_pipeline.py) ]
 ├── Train/Test split: 80% Train (171,195 rows) | 20% Test (42,582 rows), seed=42
 ├── VectorAssembler: Combines 68 features into 'raw_features' vector
 └── StandardScaler: Normalizes distributions into 'features' vector
                 │
                 ▼
[ PySpark MLlib Classifiers (train_models.py) ]
 ├── Logistic Regression
 ├── Random Forest Classifier (Top Model: 99.99% Accuracy, 0% FP)
 └── Gradient-Boosted Trees (GBT)
                 │
                 ▼
[ Interactive Streamlit Dashboard (app/app.py) ]
 ├── Overview & Telemetry Metrics
 ├── Comparative Benchmark & Confusion Matrix Viewer
 ├── Real-Time Single Flow Classifier (with 1-Click Presets)
 └── Batch Network Flow Classifier with CSV Export
```

---

## 📁 Project Directory Structure

```text
spark-network-intrusion-detection/
├── app/
│   └── app.py                      # Main Streamlit web application
├── data/
│   ├── raw/                        # Original raw CICIDS2017 CSV dataset
│   └── processed/                  # Cleaned parquet & demo CSV partitions
│       ├── demo_batch_sample.csv   # 1,000-flow balanced demo dataset (500 Benign, 500 PortScan)
│       ├── train.parquet           # 171,195 train flow records
│       └── test.parquet            # 42,582 test flow records
├── hadoop/                         # Windows winutils binaries for Spark File I/O
│   └── bin/
│       ├── hadoop.dll
│       └── winutils.exe
├── notebooks/                      # Exploratory notebooks
├── results/
│   ├── figures/                    # Saved confusion matrices & comparison charts
│   ├── models/                     # Saved PySpark MLlib models & feature pipeline
│   │   ├── feature_pipeline_model/
│   │   ├── logistic_regression_model/
│   │   ├── random_forest_model/
│   │   └── gbt_model/
│   ├── model_comparison.json       # Exact evaluation metrics
│   ├── model_comparison.csv
│   ├── model_comparison.md
│   ├── preprocessing_report.json   # Preprocessing diagnostic report
│   ├── sample_presets.json         # Preset flows for 1-click Streamlit testing
│   └── FINAL_PROJECT_REPORT.md     # Full academic project report
├── src/
│   ├── evaluation/
│   │   └── metrics.py              # Accuracy, Precision, Recall, F1, ROC-AUC, CM plots
│   ├── models/
│   │   ├── predictor.py            # Real-time and batch inference engine
│   │   └── train_models.py         # PySpark model training & evaluation runner
│   ├── preprocessing/
│   │   ├── cleaner.py              # Data cleaning and sanity filtering
│   │   ├── download_dataset.py     # Automated dataset downloader
│   │   ├── feature_pipeline.py     # VectorAssembler & StandardScaler module
│   │   ├── inspect_data.py         # Initial raw dataset diagnostics
│   │   └── run_preprocessing.py    # Master preprocessing runner
│   └── spark/
│       └── spark_session.py        # Reusable SparkSession factory with Windows bindings
├── tests/
│   ├── test_spark_env.py           # Spark smoke test
│   └── test_end_to_end.py          # Automated test suite (6 end-to-end tests)
├── .gitignore
├── README.md                       # Comprehensive project documentation
└── requirements.txt                # Python dependencies
```

---

## 🏆 Model Performance Benchmark

Evaluated on **42,582 independent test records**:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | Training Time | Test Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** ⭐ | **99.99%** | **100.00%** | **99.98%** | **99.99%** | **1.0000** | **7.01s** | 1.23s |
| **Gradient-Boosted Trees** | **99.96%** | **99.95%** | **99.96%** | **99.95%** | **0.9999** | 31.66s | **0.88s** |
| **Logistic Regression** | **99.00%** | **98.35%** | **99.34%** | **98.84%** | **0.9975** | 11.94s | 1.40s |

### Confusion Matrix Breakdown (42,582 Test Samples)

| Model | True Negatives (BENIGN) | False Positives (False Alarms) | False Negatives (Missed Attacks) | True Positives (PortScans) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **24,334** | **0** | **4** | **18,244** |
| **Gradient-Boosted Trees** | 24,324 | 10 | 8 | 18,240 |
| **Logistic Regression** | 24,030 | 304 | 120 | 18,128 |

---

## ⚙️ Installation & Setup Guide

### 1. Prerequisites
- **Python 3.12** installed and available in PATH.
- **Java OpenJDK 17 LTS**: Required for Apache Spark. If missing on Windows:
  ```powershell
  winget install Microsoft.OpenJDK.17 --source winget --accept-package-agreements --accept-source-agreements
  ```

### 2. Install Python Dependencies
From the project root folder, run:
```bash
python -m pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Launch the Streamlit Web Application
To launch the interactive dashboard, run:
```bash
python -m streamlit run app/app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

### 2. Dashboard Features & Usage
- **📊 Dashboard & Overview:** Review high-level metrics, dataset class balance, and data preprocessing impact.
- **📈 Model Performance:** Inspect side-by-side accuracy metrics, training latency, and confusion matrix heatmaps.
- **🎯 Single Flow Prediction:**
  - Select model: Random Forest, GBT, or Logistic Regression.
  - Click **"🟢 Load Benign Traffic Preset"** or **"🔴 Load PortScan Attack Preset"** to populate realistic values.
  - Click **"🚀 Analyze Network Flow with PySpark"** to view real-time intrusion verdicts, threat scores, and probability progress bars.
- **📁 Batch Prediction (CSV):**
  - Click **"🧪 Load Pre-Packaged Demo Dataset (1,000 Flows)"** or upload your own CSV file.
  - Click **"⚡ Run PySpark Batch Classification"** to process records in parallel.
  - Review traffic distributions, threat scores, and click **"💾 Download Predictions as CSV"** to save results.
- **ℹ️ About Project:** Full architecture documentation and academic references.

## 📸 Project Screenshots

### 1. Dashboard & Overview

The main dashboard provides an overview of the network intrusion detection system, including the number of raw network flows, cleaned records, engineered features, model accuracy, and dataset class distribution.

![Dashboard](screenshots/dashboard.png)

---

### 2. Model Performance

This section compares the three PySpark MLlib models used in the project: Logistic Regression, Random Forest, and Gradient-Boosted Trees. It includes accuracy, precision, recall, F1-score, ROC-AUC, training time, inference time, and confusion matrix analysis.

![Model Performance](screenshots/model_performance.png)

---

### 3. Single Flow Prediction

The Single Flow Prediction module allows a user to select a PySpark MLlib model, provide network-flow attributes, and classify an individual network flow as BENIGN or PORTSCAN.

![Single Flow Prediction](screenshots/single_prediction.png)

---

### 4. Batch Network Traffic Classification

The Batch Prediction module allows users to upload a CICIDS2017-format CSV file or use the provided demo dataset containing 1,000 network flows.

![Batch Prediction](screenshots/batch_prediction.png)

---

### 5. Batch Classification Results

After batch processing, the system displays the number of analyzed flows, benign flows, detected PortScan intrusions, the selected model, prediction results, threat scores, confidence values, and allows the predictions to be downloaded as a CSV file.

![Batch Classification Results](screenshots/batch_results.png)

### 3. Run Automated Tests
To run the automated verification suite:
```bash
python tests/test_end_to_end.py
```

### 4. (Optional) Re-execute Pipeline Stages from Scratch
If you wish to re-run data downloading, cleaning, or model training:
```bash
# 1. Inspect raw dataset
python src/preprocessing/inspect_data.py

# 2. Run data cleaning and feature scaling pipeline (Stage 3)
python src/preprocessing/run_preprocessing.py

# 3. Train and benchmark all three models (Stage 4)
python src/models/train_models.py
```

---

## 📋 Input Data Format for Batch Prediction

When uploading custom CSV files for batch classification, the file can either be:
1. **Preprocessed CSV:** Using snake_case column names (`destination_port`, `flow_duration`, etc.).
2. **Raw CICIDS2017 Format:** Original CICIDS2017 CSV columns (e.g. `' Destination Port'`, `' Flow Duration'`, `'Flow Bytes/s'`).

The pipeline's cleaning module automatically standardizes headers and fills any omitted features with default median values, ensuring zero crashes.

---

## ⚠️ Important Project Scope & Limitations

> [!IMPORTANT]
> **Scope Notice:**
> - The current trained classifiers are focused on binary detection:
>   **`BENIGN` (0) vs. `PortScan` (1)** using the CICIDS2017 PortScan benchmark partition.
> - **Boundary:** The model is specialized for detecting network port scanning, reconnaissance probing, and anomalous connection patterns. It is **not** trained to classify other attack families (e.g., SQL Injection, Cross-Site Scripting, or Infiltration) which reside in different partitions of the CICIDS2017 archive.

---

## 🔮 Future Improvements
1. **Multi-Class Expansion:** Incorporate additional partitions from CICIDS2017 (DDoS, Botnet, Web Attacks) into a unified multi-class classifier.
2. **Spark Structured Streaming:** Ingest real-time packet capture streams via Apache Kafka into a Spark Structured Streaming window for live intrusion alerting.
3. **Unsupervised Anomaly Detection:** Deploy Spark MLlib Isolation Forests or Autoencoders for zero-day threat discovery.
