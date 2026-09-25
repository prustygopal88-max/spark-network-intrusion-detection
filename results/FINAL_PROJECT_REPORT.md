# Scalable Network Intrusion Detection and Anomaly Analysis Using Apache Spark

**College Mini-Project Final Technical Report**  
*Academic Year: 2026*  
*Technology Stack: Python 3.12 | Apache Spark 3.5.9 | PySpark MLlib | OpenJDK 17 LTS | Streamlit 1.44*

---

## 1. Abstract
The exponential increase in high-speed enterprise computer network traffic has rendered conventional, single-threaded Intrusion Detection Systems (NIDS) incapable of real-time telemetry processing without packet drop. This project presents a scalable, distributed Big Data network intrusion detection system built upon **Apache Spark** and **PySpark MLlib**. Using the benchmark **CICIDS2017** dataset from the Canadian Institute for Cybersecurity, we engineered an end-to-end distributed data engineering and machine learning pipeline. The pipeline handles data hygiene, infinite value sanitization, deduplication, feature scaling (`StandardScaler`), vector assembly (`VectorAssembler`), and trains three distinct distributed classifiers: **Logistic Regression**, **Random Forest**, and **Gradient-Boosted Trees (GBT)**. On a test set of 42,582 network flows, the PySpark Random Forest model attained **99.99% accuracy**, **100% precision**, and **0.9999 F1-score** with **zero false alarms (0 False Positives)** out of 24,334 benign samples. A responsive web dashboard developed in **Streamlit** provides real-time single-flow prediction, batch flow classification, and comparative performance visualization.

---

## 2. Introduction
Modern network infrastructures are persistently targeted by malicious scanning, probes, denial-of-service, and reconnaissance tactics. A crucial precursor to cyberattacks is the **PortScan**, where adversaries probe active ports to identify vulnerabilities and running services. Traditional machine learning pipelines using single-node environments (e.g., standard Scikit-Learn in memory) encounter out-of-memory bottlenecks when scaling to millions of flow records. **Apache Spark** addresses this through distributed Resilient Distributed Datasets (RDDs) and Catalyst Optimizer-backed DataFrames, enabling linear scaling across multi-core systems and server clusters.

---

## 3. Problem Statement
To design, implement, and evaluate an automated, scalable network flow intrusion detection architecture capable of:
1. Ingesting large volumes of high-dimensional network flow records without manual feature parsing.
2. Sanitizing division-by-zero flow anomalies, handling duplicates, and eliminating non-informative features.
3. Training distributed machine learning models via PySpark MLlib that maximize intrusion detection recall while minimizing false alarm rates.
4. Providing an intuitive, interactive web interface for network security analysts to perform single-flow telemetry inspection and bulk CSV classification.

---

## 4. Project Objectives
- **Data Engineering:** Construct a robust PySpark preprocessing pipeline to clean, standardize, and scale flow features.
- **Model Training:** Train and benchmark Logistic Regression, Random Forest, and Gradient-Boosted Trees using PySpark MLlib.
- **Performance Evaluation:** Rigorously evaluate models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices on unseen test data.
- **Deployment & Visualization:** Deploy an interactive Streamlit web dashboard allowing security personnel to simulate live traffic and analyze bulk flow logs.

---

## 5. Dataset Description
The system utilizes the **CICIDS2017** dataset developed by the Canadian Institute for Cybersecurity (CIC) and University of New Brunswick (UNB).
- **Partition:** `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- **Raw Volume:** 286,467 network flows (73.34 MB)
- **Features:** 78 flow metrics (Flow Duration, Inter-Arrival Times, Packet Length Statistics, TCP Flags, Header Lengths, Subflow Stats) + 1 Label column.
- **Target Classes:**
  - `BENIGN`: Normal background business traffic (HTTP, SSH, DNS, etc.)
  - `PortScan`: Systematic TCP/UDP port reconnaissance probe traffic

---

## 6. System Architecture & Methodology

```text
[ Raw CICIDS2017 CSV Telemetry ]
               │
               ▼
[ Preprocessing & Data Cleaning Module ]
 ├── Trim Column Names & Snake_Case Conversion
 ├── Encode Target: BENIGN -> 0, PortScan -> 1
 ├── Filter Duplicates (72,353 rows dropped)
 ├── Sanitize Inf & Nulls (337 rows dropped)
 └── Prune Zero-Variance Constant Features (10 columns dropped)
               │
               ▼
[ Cleaned Dataset: 213,777 rows, 68 predictive features ]
               │
               ▼
[ PySpark Distributed Engine (local[*], OpenJDK 17 LTS) ]
 ├── Train / Test Split (80% Train: 171,195 | 20% Test: 42,582, seed=42)
 ├── PySpark VectorAssembler (68 features -> raw_features vector)
 └── PySpark StandardScaler (raw_features -> scaled features vector)
               │
               ▼
[ Spark MLlib Model Training & Evaluation ]
 ├── Logistic Regression (Baseline Linear Model)
 ├── Random Forest (Ensemble Distributed Bagging)
 └── Gradient-Boosted Trees (Ensemble Sequential Boosting)
               │
               ▼
[ Interactive Streamlit Web Application ]
 ├── Executive Dashboard & Telemetry Metrics
 ├── Comparative Benchmark & Confusion Matrix Inspector
 ├── Real-Time Single Flow Classifier (with 1-Click Presets)
 └── Batch Network Flow Classifier with CSV Export
```

---

## 7. Data Preprocessing & Feature Engineering

1. **Duplicate Elimination:** 72,353 identical network flow logs were identified and removed. In network captures, repeated zero-window SYN retries produce redundant rows that skew model evaluation.
2. **Infinite Value Treatment:** In high-speed captures, near-zero flow durations cause division-by-zero when computing `Flow Bytes/s` and `Flow Packets/s`, generating $\pm\infty$. These were replaced with `NaN` and the 337 affected rows were dropped.
3. **Zero-Variance Feature Pruning:** 10 features were identified as completely constant across all flows (`bwd_psh_flags`, `fwd_urg_flags`, `bwd_urg_flags`, `cwe_flag_count`, `fwd_avg_bytes_bulk`, `fwd_avg_packets_bulk`, `fwd_avg_bulk_rate`, `bwd_avg_bytes_bulk`, `bwd_avg_packets_bulk`, `bwd_avg_bulk_rate`). Dropping these prevented division-by-zero during standard scaling and saved memory.
4. **Strict Isolation:** To prevent data leakage, the Spark `Pipeline` (`VectorAssembler` + `StandardScaler`) was fitted **only on the 80% training partition**, and then used to transform both train and test partitions.

---

## 8. Experimental Results & Model Comparison

Evaluated on **42,582 independent test records**:

| Metric | Logistic Regression | Random Forest ⭐ | Gradient-Boosted Trees |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.00% | **99.99%** | 99.96% |
| **Precision** | 98.35% | **100.00%** | 99.95% |
| **Recall** | 99.34% | **99.98%** | 99.96% |
| **F1-Score** | 98.84% | **99.99%** | 99.95% |
| **ROC-AUC** | 0.9975 | **1.0000** | 0.9999 |
| **Training Time** | 11.94s | **7.01s** | 31.66s |
| **Inference Time (42k rows)**| 1.40s | 1.23s | **0.88s** |

### Confusion Matrix Breakdown

| Metric | Logistic Regression | Random Forest | Gradient-Boosted Trees |
| :--- | :---: | :---: | :---: |
| **True Negatives (BENIGN)** | 24,030 | **24,334** | 24,324 |
| **False Positives (False Alarms)** | 304 | **0** | 10 |
| **False Negatives (Missed Attacks)**| 120 | **4** | 8 |
| **True Positives (PortScans)** | 18,128 | **18,244** | 18,240 |

---

## 9. Key Findings & Discussion
- **Superiority of Random Forest:** Spark MLlib's parallelized Random Forest emerged as the top performer. It attained **100% precision with 0 False Positives**, which is essential for SOC (Security Operations Center) environments to eliminate alert fatigue.
- **Fast Training:** Due to Spark's tree bagging parallelization, Random Forest trained in only **7.01 seconds** across 171k rows.
- **GBT Latency:** Gradient-Boosted Trees offered the lowest inference latency (0.88 seconds) but required longer sequential training (31.66s).

---

## 10. Streamlit Web Interface
The web application provides five core pages:
1. **Dashboard & Overview:** Visual summaries of dataset volume, class breakdown, and preprocessing impact.
2. **Model Performance:** Side-by-side metric tables, comparison bar charts, and an interactive confusion matrix viewer.
3. **Single Flow Prediction:** Allows security analysts to input packet parameters, choose any of the three models, or load 1-click **Benign** or **PortScan** demo presets.
4. **Batch Prediction:** Supports uploading external network flow CSV files or running the built-in 1,000-flow test dataset, outputting predictions, confidence scores, and downloadable CSV reports.
5. **About Project:** Project metadata, technology details, and methodology notes.

---

## 11. Important Project Scope & Limitations
> [!IMPORTANT]
> **Project Limitation Notice:**
> The current trained models are specifically developed and evaluated for binary network classification:
> **`BENIGN` (0) vs. `PortScan` (1)** using the CICIDS2017 PortScan benchmark partition.
> 
> - **Scope:** This system is optimized for detecting network scanning, port probing, and reconnaissance flows against normal background traffic.
> - **Boundary:** The model is **not** trained to classify other attack families (e.g., SQL Injection, Cross-Site Scripting, Infiltration, or Heartbleed) that are contained in separate PCAP partitions of the full CICIDS2017 archive. Any generalization claims should be understood within the context of network flow-based reconnaissance and anomaly detection.

---

## 12. Conclusion & Future Scope
This project demonstrated that combining Apache Spark's distributed data pipeline with PySpark MLlib enables high-accuracy (99.99%), low-latency network intrusion detection on large-scale network telemetry.

### Future Scope:
1. **Multi-Class Expansion:** Incorporate additional CICIDS2017 files (DDoS, Botnet, Web Attacks) into a multi-class intrusion classification pipeline.
2. **Spark Structured Streaming:** Integrate Apache Kafka to feed live packet flows directly into Spark Structured Streaming for real-time sliding-window anomaly detection.
3. **Deep Learning on Spark:** Experiment with PySpark Deep Learning pipelines (e.g., Spark-Torch) using autoencoders for unsupervised zero-day anomaly detection.
