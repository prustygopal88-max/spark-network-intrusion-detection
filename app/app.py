import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# Setup paths and environment
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

# Page Configuration
st.set_page_config(
    page_title="Spark NIDS - Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished college presentation UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-benign {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #31C48D;
    }
    .badge-attack {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #F98080;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load metadata and comparison data
@st.cache_data
def load_project_metadata():
    prep_report_path = os.path.join(PROJECT_ROOT, "results", "preprocessing_report.json")
    model_report_path = os.path.join(PROJECT_ROOT, "results", "model_comparison.json")
    presets_path = os.path.join(PROJECT_ROOT, "results", "sample_presets.json")

    prep_data = {}
    model_data = []
    presets = {}

    if os.path.exists(prep_report_path):
        with open(prep_report_path, "r") as f:
            prep_data = json.load(f)

    if os.path.exists(model_report_path):
        with open(model_report_path, "r") as f:
            model_data = json.load(f)

    if os.path.exists(presets_path):
        with open(presets_path, "r") as f:
            presets = json.load(f)

    return prep_data, model_data, presets

# Helper: Cached Spark Artifacts
@st.cache_resource(show_spinner="Initializing Apache Spark cluster runtime & loading MLlib models...")
def get_cached_spark_engine():
    return load_inference_artifacts()

# Load metadata
prep_meta, model_results, sample_presets = load_project_metadata()

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.image("https://spark.apache.org/images/spark-logo-trademark.png", width=190)
st.sidebar.markdown("### **Navigation**")
nav_choice = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard & Overview",
        "📈 Model Performance",
        "🎯 Single Flow Prediction",
        "📁 Batch Prediction (CSV)",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### **System Environment**")
st.sidebar.info(
    """
    - **Engine:** Apache Spark 3.5.9
    - **Runtime:** OpenJDK 17 LTS
    - **Interface:** PySpark MLlib
    - **Dataset:** CICIDS2017
    - **Target:** BENIGN (0) vs PortScan (1)
    """
)
st.sidebar.caption("Scalable Network Intrusion Detection System")

# ==============================================================================
# VIEW 1: DASHBOARD & OVERVIEW
# ==============================================================================
if nav_choice == "📊 Dashboard & Overview":
    st.markdown('<div class="main-title">🛡️ Scalable Network Intrusion Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Big Data Network Traffic Anomaly & Attack Analysis using <b>Apache Spark</b> & <b>PySpark MLlib</b></div>', unsafe_allow_html=True)

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Raw Network Flows", value=f"{prep_meta.get('raw_rows', 286467):,}")
    with col2:
        st.metric(label="Cleaned Flow Records", value=f"{prep_meta.get('cleaned_rows', 213777):,}")
    with col3:
        st.metric(label="Engineered Features", value=f"{prep_meta.get('feature_count', 68)}")
    with col4:
        st.metric(label="Top Model Accuracy", value="99.99%", delta="Random Forest")

    st.markdown("---")

    # Overview Cards
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.subheader("📌 Project Motivation & Architecture")
        st.markdown(
            """
            Modern enterprise computer networks handle gigabytes of packet transactions every second. 
            Traditional single-threaded intrusion detection systems fail to scale under high-throughput traffic.
            
            This project implements an **enterprise-grade, distributed Network Intrusion Detection System (NIDS)** 
            built upon **Apache Spark**, designed to analyze high-speed network telemetry flows in parallel:
            
            1. **Dataset:** Official **CICIDS2017** benchmark from the Canadian Institute for Cybersecurity.
            2. **Distributed Engine:** **Apache Spark** using multi-core distributed in-memory compute.
            3. **Feature Engineering:** Spark MLlib `VectorAssembler` and `StandardScaler` normalizing 68 network metrics.
            4. **Classification:** High-precision supervised detection distinguishing legitimate background flows from malicious port probes.
            """
        )

    with c2:
        st.subheader("📊 Dataset Class Distribution")
        clean_counts = prep_meta.get("class_distribution_after", {}).get("counts", {"BENIGN (0)": 123083, "PortScan (1)": 90694})
        pie_df = pd.DataFrame({
            "Traffic Type": list(clean_counts.keys()),
            "Records": list(clean_counts.values())
        })
        st.bar_chart(pie_df.set_index("Traffic Type"), color="#2563EB")
        st.caption(f"BENIGN: {clean_counts.get('BENIGN (0)', 0):,} (57.6%) | PortScan: {clean_counts.get('PortScan (1)', 0):,} (42.4%)")

    st.markdown("---")
    st.subheader("🧹 Spark Preprocessing Pipeline Impact")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(f"""
        **Duplicate Removal**
        - Duplicate rows removed: **{prep_meta.get('duplicates_removed', 72353):,}**
        - Eliminates zero-length probe retransmissions.
        """)
    with colB:
        st.markdown(rf"""
        **Anomaly & Infinite Value Cleaning**
        - Rows with $\pm\infty$ or NaN removed: **{prep_meta.get('missing_and_inf_rows_removed', 337):,}**
        - Solves zero-duration division anomalies in `flow_bytes_s`.
        """)
    with colC:
        st.markdown(f"""
        **Zero-Variance Feature Pruning**
        - Constant columns dropped: **{len(prep_meta.get('constant_columns_dropped', []))}**
        - Removes inactive TCP flag features with zero variance.
        """)

# ==============================================================================
# VIEW 2: MODEL PERFORMANCE
# ==============================================================================
elif nav_choice == "📈 Model Performance":
    st.markdown('<div class="main-title">📈 PySpark MLlib Model Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparative evaluation across <b>42,582</b> held-out test network flow records</div>', unsafe_allow_html=True)

    # Performance Table
    if model_results:
        table_rows = []
        for r in model_results:
            cm = r["confusion_matrix"]
            table_rows.append({
                "Model": r["model_name"],
                "Accuracy (%)": f"{r['accuracy'] * 100:.2f}%",
                "Precision (%)": f"{r['precision'] * 100:.2f}%",
                "Recall (%)": f"{r['recall'] * 100:.2f}%",
                "F1-Score (%)": f"{r['f1_score'] * 100:.2f}%",
                "ROC-AUC": f"{r['roc_auc']:.4f}",
                "Train Time": f"{r['train_time_sec']}s",
                "Inference Time": f"{r['inference_time_sec']}s",
                "True Negatives": f"{cm['tn']:,}",
                "False Positives": f"{cm['fp']:,}",
                "False Negatives": f"{cm['fn']:,}",
                "True Positives": f"{cm['tp']:,}"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # Comparison Plots
    col_chart, col_cm = st.columns([1.1, 0.9])
    
    with col_chart:
        st.subheader("📊 Comparative Evaluation Chart")
        comp_img = os.path.join(PROJECT_ROOT, "results", "figures", "model_comparison_bar.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Accuracy, Precision, Recall, and F1 Comparison", use_container_width=True)
        else:
            st.info("Comparison chart available in results/figures/.")

    with col_cm:
        st.subheader("🎯 Confusion Matrix Inspector")
        model_pick = st.selectbox(
            "Select model to inspect confusion matrix:",
            ["Random Forest", "Gradient-Boosted Trees", "Logistic Regression"]
        )
        file_tag = model_pick.lower().replace(" ", "_").replace("-", "_")
        cm_file = os.path.join(PROJECT_ROOT, "results", "figures", f"{file_tag}_cm.png")
        if not os.path.exists(cm_file):
            # Fallback filename check
            if "boost" in file_tag:
                cm_file = os.path.join(PROJECT_ROOT, "results", "figures", "gradient-boosted_trees_cm.png")
        
        if os.path.exists(cm_file):
            st.image(cm_file, caption=f"Confusion Matrix: {model_pick}", use_container_width=True)
        else:
            st.warning(f"Plot not found at: {cm_file}")

    st.markdown("---")
    st.subheader("💡 Technical Assessment & Discussion")
    st.markdown(
        """
        - **Random Forest (Best Performing):** Achieved **99.99% accuracy** with **0 False Positives** across 24,334 benign samples. In cyber-defense environments, zero false alarms prevents alert fatigue for SOC analysts.
        - **Gradient-Boosted Trees:** Exceptional accuracy (99.96%) with sub-second inference (0.88s) across 42k records.
        - **Logistic Regression:** High baseline accuracy (99.00%) with near-zero memory footprint, but higher false alarm rates on non-linear boundary cases.
        """
    )

# ==============================================================================
# VIEW 3: SINGLE FLOW PREDICTION
# ==============================================================================
elif nav_choice == "🎯 Single Flow Prediction":
    st.markdown('<div class="main-title">🎯 Real-Time Network Flow Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Simulate incoming network telemetry and classify intrusion threats using PySpark MLlib</div>', unsafe_allow_html=True)

    # Initialize Spark models
    with st.spinner("Connecting to PySpark execution engine..."):
        get_cached_spark_engine()

    col_model, col_presets = st.columns([1, 1.2])
    with col_model:
        selected_model = st.selectbox(
            "Select PySpark ML Model:",
            ["Random Forest", "Gradient-Boosted Trees", "Logistic Regression"],
            help="Choose the trained MLlib model used to classify this network flow."
        )

    with col_presets:
        st.markdown("**Quick Preset Buttons:**")
        p_col1, p_col2 = st.columns(2)
        preset_choice = None
        with p_col1:
            if st.button("🟢 Load Benign Traffic Preset", use_container_width=True):
                st.session_state["flow_inputs"] = sample_presets.get("BENIGN", {})
                st.rerun()
        with p_col2:
            if st.button("🔴 Load PortScan Attack Preset", use_container_width=True):
                st.session_state["flow_inputs"] = sample_presets.get("PORTSCAN", {})
                st.rerun()

    # Load current inputs from session_state or defaults
    current_inputs = st.session_state.get("flow_inputs", sample_presets.get("BENIGN", {}))

    st.markdown("---")
    st.subheader("⚙️ Key Network Flow Attributes")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dest_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=int(current_inputs.get("destination_port", 80)))
        flow_duration = st.number_input("Flow Duration (µs)", min_value=0.0, value=float(current_inputs.get("flow_duration", 1000000.0)), step=1000.0)
    with c2:
        tot_fwd_pkts = st.number_input("Total Fwd Packets", min_value=0.0, value=float(current_inputs.get("total_fwd_packets", 5.0)))
        tot_bwd_pkts = st.number_input("Total Bwd Packets", min_value=0.0, value=float(current_inputs.get("total_backward_packets", 4.0)))
    with c3:
        pkt_len_mean = st.number_input("Packet Length Mean", min_value=0.0, value=float(current_inputs.get("packet_length_mean", 120.0)))
        pkt_len_std = st.number_input("Packet Length Std", min_value=0.0, value=float(current_inputs.get("packet_length_std", 50.0)))
    with c4:
        flow_bytes_s = st.number_input("Flow Bytes/sec", min_value=0.0, value=float(current_inputs.get("flow_bytes_s", 2500.0)))
        init_win_fwd = st.number_input("Init Win Bytes Fwd", min_value=0.0, value=float(current_inputs.get("init_win_bytes_forward", 8192.0)))

    # Advanced features accordion
    with st.expander("🛠️ Advanced Flow Features (68 Features Total)"):
        st.caption("All features are automatically mapped to the PySpark VectorAssembler and StandardScaler.")
        adv_cols = st.columns(4)
        all_features = get_feature_columns()
        dynamic_inputs = dict(current_inputs)

        # Update core features
        dynamic_inputs["destination_port"] = dest_port
        dynamic_inputs["flow_duration"] = flow_duration
        dynamic_inputs["total_fwd_packets"] = tot_fwd_pkts
        dynamic_inputs["total_backward_packets"] = tot_bwd_pkts
        dynamic_inputs["packet_length_mean"] = pkt_len_mean
        dynamic_inputs["packet_length_std"] = pkt_len_std
        dynamic_inputs["flow_bytes_s"] = flow_bytes_s
        dynamic_inputs["init_win_bytes_forward"] = init_win_fwd

        for i, col_name in enumerate(all_features):
            if col_name not in ["destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets", "packet_length_mean", "packet_length_std", "flow_bytes_s", "init_win_bytes_forward"]:
                with adv_cols[i % 4]:
                    default_val = float(dynamic_inputs.get(col_name, 0.0))
                    dynamic_inputs[col_name] = st.number_input(col_name, value=default_val, key=f"adv_{col_name}")

    st.markdown("")
    if st.button("🚀 Analyze Network Flow with PySpark", type="primary", use_container_width=True):
        try:
            with st.spinner("Executing Spark MLlib VectorAssembler and model inference..."):
                res = predict_single_flow(dynamic_inputs, model_name=selected_model)

            st.markdown("---")
            st.subheader("🔍 Intrusion Detection Result")
            res_col1, res_col2 = st.columns([1.2, 0.8])

            with res_col1:
                if res["prediction_label"] == "PORTSCAN":
                    st.markdown('<div class="badge-attack">🚨 MALICIOUS INTRUSION: PORTSCAN DETECTED</div>', unsafe_allow_html=True)
                    st.error(f"**Threat Alert:** Port scanning activity identified with **{res['confidence'] * 100:.2f}% confidence** using {selected_model}.")
                else:
                    st.markdown('<div class="badge-benign">✅ NORMAL NETWORK TRAFFIC (BENIGN)</div>', unsafe_allow_html=True)
                    st.success(f"**Safe:** Normal flow transaction verified with **{res['confidence'] * 100:.2f}% confidence** using {selected_model}.")

            with res_col2:
                st.markdown("**Model Class Probabilities:**")
                st.progress(res["portscan_prob"], text=f"PortScan Threat Probability: {res['portscan_prob'] * 100:.2f}%")
                st.progress(res["benign_prob"], text=f"Benign Probability: {res['benign_prob'] * 100:.2f}%")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ==============================================================================
# VIEW 4: BATCH PREDICTION (CSV)
# ==============================================================================
elif nav_choice == "📁 Batch Prediction (CSV)":
    st.markdown('<div class="main-title">📁 Batch Network Traffic Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a packet capture flow CSV or use the pre-packaged balanced dataset for bulk classification</div>', unsafe_allow_html=True)

    # Initialize Spark
    with st.spinner("Connecting to Spark cluster backend..."):
        get_cached_spark_engine()

    col_b_model, col_b_sample = st.columns([1, 1])
    with col_b_model:
        batch_model = st.selectbox(
            "Select PySpark Batch Model:",
            ["Random Forest", "Gradient-Boosted Trees", "Logistic Regression"]
        )

    with col_b_sample:
        demo_file_path = os.path.join(PROJECT_ROOT, "data", "processed", "demo_batch_sample.csv")
        load_demo = st.button("🧪 Load Pre-Packaged Demo Dataset (1,000 Flows)", use_container_width=True)

    uploaded_file = st.file_uploader("Upload Network Flow CSV (CICIDS2017 format or preprocessed):", type=["csv"])

    input_df = None
    source_name = ""

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file, low_memory=False)
            source_name = uploaded_file.name
            st.success(f"Uploaded `{source_name}` successfully ({len(input_df):,} records).")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    elif load_demo and os.path.exists(demo_file_path):
        input_df = pd.read_csv(demo_file_path)
        source_name = "demo_batch_sample.csv"
        st.info(f"Loaded demo dataset: `{source_name}` (500 BENIGN + 500 PortScan records).")

    if input_df is not None:
        st.markdown(f"**Data Preview (`{source_name}`):**")
        st.dataframe(input_df.head(5), use_container_width=True)

        if st.button("⚡ Run PySpark Batch Classification", type="primary"):
            try:
                with st.spinner(f"Processing {len(input_df):,} records through Spark MLlib {batch_model}..."):
                    results_df, summary = predict_batch_flows(input_df, model_name=batch_model)

                st.markdown("---")
                st.subheader("📊 Batch Classification Summary")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Flows Analyzed", f"{summary['total_analyzed']:,}")
                with m2:
                    st.metric("Benign Flows", f"{summary['benign_count']:,}", f"{summary['benign_pct']}%")
                with m3:
                    st.metric("PortScan Intrusions", f"{summary['portscan_count']:,}", f"-{summary['portscan_pct']}%", delta_color="inverse")
                with m4:
                    st.metric("Model Used", summary["model_used"])

                # Distribution Chart
                summary_chart_df = pd.DataFrame({
                    "Traffic Type": ["BENIGN", "PORTSCAN"],
                    "Count": [summary["benign_count"], summary["portscan_count"]]
                })
                st.bar_chart(summary_chart_df.set_index("Traffic Type"), color="#DC2626")

                # Results Table
                st.markdown("### 📋 Prediction Results (First 500 rows)")
                display_cols = ["Predicted_Class", "Threat_Score (%)", "Confidence (%)"] + [c for c in results_df.columns if c not in ["Predicted_Class", "Threat_Score (%)", "Confidence (%)"]][:6]
                st.dataframe(results_df[display_cols].head(500), use_container_width=True)

                # CSV Download Button
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Predictions as CSV",
                    data=csv_bytes,
                    file_name=f"pyspark_nids_predictions_{batch_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Batch processing error: {e}")

# ==============================================================================
# VIEW 5: ABOUT PROJECT
# ==============================================================================
elif nav_choice == "ℹ️ About Project":
    st.markdown('<div class="main-title">ℹ️ Project Documentation & Specifications</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Scalable Network Intrusion Detection and Anomaly Analysis Using Apache Spark</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 🎓 Academic Project Overview
        - **Title:** Scalable Network Intrusion Detection and Anomaly Analysis Using Apache Spark
        - **Domain:** Big Data Analytics, Cybersecurity, Distributed Machine Learning
        - **Dataset:** Official **CICIDS2017** Network Traffic Intrusion Benchmark
        
        ### 🛠️ Technology Stack
        - **Distributed Framework:** Apache Spark 3.5.9
        - **ML Library:** Spark MLlib (`VectorAssembler`, `StandardScaler`, `RandomForestClassifier`, `GBTClassifier`, `LogisticRegression`)
        - **Runtime:** Microsoft OpenJDK 17 LTS (JVM)
        - **Programming Language:** Python 3.12
        - **Web Dashboard:** Streamlit 1.44
        - **Data Handling:** Pandas 2.2, NumPy 2.2, Scikit-learn 1.8
        
        ### 📋 Preprocessing Decisions
        1. **Feature Pruning:** Removed 10 constant features having zero variance.
        2. **Anomaly Elimination:** Detected and sanitized 659 infinite values in division-by-zero flow calculations.
        3. **Deduplication:** Filtered 72,353 identical network logs.
        4. **Class Mapping:** `BENIGN` $\\rightarrow 0$, `PortScan` $\\rightarrow 1$.
        5. **Strict Train/Test Isolation:** Feature pipeline fitted only on the 80% training partition (`seed=42`) to prevent data leakage.
        
        ### 🏆 Key Results
        - **Top Accuracy:** **99.99%** achieved with PySpark Random Forest.
        - **False Alarm Rate:** **0% (0 False Positives)** out of 24,334 benign samples.
        - **Inference Speed:** Average 1.1s across 42,000 records on multi-core Spark.
        """
    )
