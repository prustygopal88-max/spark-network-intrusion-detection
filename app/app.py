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
from src.evaluation.xai_explainer import explain_prediction

# Page Configuration
st.set_page_config(
    page_title="NetGuard AI - Scalable Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End SaaS CSS Styling (Enterprise Cybersecurity Palette)
st.markdown("""
<style>
    /* Global Typography & Backgrounds */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        padding: 32px 36px;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 0px;
    }
    .role-badge-user {
        background-color: #0D9488;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
    }
    .role-badge-admin {
        background-color: #4F46E5;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
    }
    
    /* Elegant Information & XAI Cards */
    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .xai-card-attack {
        background-color: #FEF2F2;
        border: 1.5px solid #F87171;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.08);
    }
    .xai-card-benign {
        background-color: #F0FDF4;
        border: 1.5px solid #4ADE80;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        box-shadow: 0 4px 14px rgba(34, 197, 94, 0.08);
    }
    
    .verdict-banner-attack {
        background: linear-gradient(90deg, #DC2626 0%, #991B1B 100%);
        color: #FFFFFF;
        padding: 16px 20px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: -0.3px;
        text-align: center;
        margin-bottom: 14px;
    }
    .verdict-banner-benign {
        background: linear-gradient(90deg, #16A34A 0%, #166534 100%);
        color: #FFFFFF;
        padding: 16px 20px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: -0.3px;
        text-align: center;
        margin-bottom: 14px;
    }
    
    .step-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        height: 100%;
    }
    .step-num {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0F766E;
        margin-bottom: 4px;
    }
    .step-title {
        font-weight: 700;
        font-size: 1.0rem;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .step-desc {
        font-size: 0.88rem;
        color: #64748B;
        line-height: 1.4;
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
@st.cache_resource(show_spinner="Initializing Apache Spark Cluster & MLlib Models...")
def get_cached_spark_engine():
    return load_inference_artifacts()

prep_meta, model_results, sample_presets = load_project_metadata()

# ==============================================================================
# SIDEBAR: ROLE SELECTOR & ACCESS GATEWAY
# ==============================================================================
st.sidebar.image("https://spark.apache.org/images/spark-logo-trademark.png", width=190)
st.sidebar.markdown("### **Portal Selection**")

portal_choice = st.sidebar.radio(
    "Choose View Portal:",
    [
        "👤 Guest / Live Demo (Non-Technical)",
        "🛡️ Admin & Engineer Portal (Deep Tech)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### **System Environment**")
st.sidebar.caption("• **Engine:** Apache Spark 3.5.9")
st.sidebar.caption("• **Runtime:** OpenJDK 17 LTS")
st.sidebar.caption("• **ML Library:** PySpark MLlib")
st.sidebar.caption("• **Dataset:** CICIDS2017 Benchmark")
st.sidebar.caption("• **Target:** BENIGN (0) vs PortScan (1)")

# ==============================================================================
# PORTAL 1: GUEST / LIVE DEMO PORTAL (NON-TECHNICAL & EXPLAINABLE AI)
# ==============================================================================
if "Guest" in portal_choice:
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🛡️ NetGuard AI: Smart Intrusion Detection</div>
        <div class="hero-subtitle">Explainable Artificial Intelligence (XAI) for Network Security, Powered by <b>Apache Spark</b></div>
        <span class="role-badge-user">👤 Guest / Non-Technical Demo Mode Active</span>
    </div>
    """, unsafe_allow_html=True)

    # Educational Walkthrough (What is this website used for?)
    with st.container():
        st.markdown("""
        <div class="info-card">
            <h3 style="margin-top:0; color:#0F172A; font-weight:700;">📖 What is this website used for?</h3>
            <p style="color:#475569; font-size:1.0rem; line-height:1.6; margin-bottom:14px;">
                This website acts as an <b>intelligent digital security guard</b> for computer networks. 
                Just like an airport scanner checks luggage for dangerous items without slowing travelers down, 
                our <b>Apache Spark AI system</b> continuously inspects thousands of internet connections in real time 
                to catch hackers who are secretly scanning company servers (known as <b>PortScan attacks</b>) before they can break in.
            </p>
            <h4 style="margin-bottom:12px; color:#1E293B;">🔍 How does the AI work in 3 simple steps?</h4>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("""
            <div class="step-box">
                <div class="step-num">Step 1</div>
                <div class="step-title">Capture Network Flow</div>
                <div class="step-desc">When computers communicate, we measure connection speed, packet size, and targeted ports.</div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown("""
            <div class="step-box">
                <div class="step-num">Step 2</div>
                <div class="step-title">Spark AI Evaluation</div>
                <div class="step-desc">Apache Spark's Random Forest model evaluates 68 behavioral safety markers in milliseconds.</div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            st.markdown("""
            <div class="step-box">
                <div class="step-num">Step 3</div>
                <div class="step-title">Reason-Based XAI Verdict</div>
                <div class="step-desc">The AI gives a plain-English explanation of exactly WHY the traffic is safe or suspicious.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Demo Tabs: Single Flow vs Batch CSV
    demo_tab1, demo_tab2 = st.tabs(["🎯 Live Flow Scanner (Single Flow)", "📁 Bulk Traffic Scanner (CSV File)"])

    # --------------------------------------------------------------------------
    # DEMO TAB 1: SINGLE FLOW PREDICTION + REASON-BASED XAI
    # --------------------------------------------------------------------------
    with demo_tab1:
        st.subheader("🎯 Test a Live Network Connection")
        st.write("Click a preset below to test an authentic safe connection or a hacker's attack probe:")

        # Initialize Spark engine
        get_cached_spark_engine()

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🟢 Load Safe Web Browsing Sample", use_container_width=True):
                st.session_state["guest_flow"] = sample_presets.get("BENIGN", {})
                st.rerun()
        with btn_col2:
            if st.button("🔴 Load Hacker PortScan Attack Sample", use_container_width=True):
                st.session_state["guest_flow"] = sample_presets.get("PORTSCAN", {})
                st.rerun()

        current_flow = st.session_state.get("guest_flow", sample_presets.get("BENIGN", {}))

        st.markdown("---")
        st.markdown("##### 🎛️ Adjust Key Telemetry (Or leave as loaded from preset):")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            dest_port = st.number_input("Target Port (e.g. 80=Web, 22=SSH)", min_value=0, max_value=65535, value=int(current_flow.get("destination_port", 80)))
            flow_duration = st.number_input("Connection Duration (microseconds)", min_value=0.0, value=float(current_flow.get("flow_duration", 1000000.0)), step=10000.0)
        with fc2:
            fwd_pkts = st.number_input("Packets Sent by Sender", min_value=0.0, value=float(current_flow.get("total_fwd_packets", 5.0)))
            bwd_pkts = st.number_input("Packets Answered by Server", min_value=0.0, value=float(current_flow.get("total_backward_packets", 4.0)))
        with fc3:
            pkt_len_mean = st.number_input("Average Data Size per Packet (bytes)", min_value=0.0, value=float(current_flow.get("packet_length_mean", 120.0)))
            flow_bytes_s = st.number_input("Data Transfer Rate (bytes/sec)", min_value=0.0, value=float(current_flow.get("flow_bytes_s", 2500.0)))

        # Update dictionary
        input_data = dict(current_flow)
        input_data["destination_port"] = dest_port
        input_data["flow_duration"] = flow_duration
        input_data["total_fwd_packets"] = fwd_pkts
        input_data["total_backward_packets"] = bwd_pkts
        input_data["packet_length_mean"] = pkt_len_mean
        input_data["flow_bytes_s"] = flow_bytes_s

        if st.button("🚀 Scan Connection with Spark AI", type="primary", use_container_width=True):
            with st.spinner("Analyzing 68 flow markers using Apache Spark Random Forest model..."):
                raw_pred = predict_single_flow(input_data, model_name="Random Forest")
                xai = explain_prediction(input_data, raw_pred)

            st.markdown("---")
            st.subheader("🔍 AI Security Verdict & Explainable AI (XAI) Report")

            if raw_pred["prediction_label"] == "PORTSCAN":
                st.markdown(f'<div class="verdict-banner-attack">🚨 THREAT DETECTED: MALICIOUS PORT SCAN ({xai["confidence_pct"]}% Confidence)</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="xai-card-attack">
                    <h3 style="color:#991B1B; margin-top:0;">🧠 Plain-English AI Explanation: Why is this Suspicious?</h3>
                    <p style="font-size:1.05rem; color:#7F1D1D; line-height:1.6; font-weight:500;">
                        {xai['layman_summary']}
                    </p>
                    <hr style="border:0.5px solid #FCA5A5; margin:16px 0;">
                    <h4 style="color:#991B1B;">📋 Key Red Flags Identified by the Model:</h4>
                """, unsafe_allow_html=True)
                for r in xai["reasons"]:
                    st.markdown(f"- {r}")
                st.markdown(f"""
                    <div style="background-color:#FEE2E2; padding:12px; border-radius:8px; margin-top:14px; color:#991B1B; font-weight:600;">
                        🛡️ {xai['recommended_action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f'<div class="verdict-banner-benign">✅ VERIFIED SAFE: NORMAL NETWORK TRAFFIC ({xai["confidence_pct"]}% Confidence)</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="xai-card-benign">
                    <h3 style="color:#166534; margin-top:0;">🧠 Plain-English AI Explanation: Why is this Safe?</h3>
                    <p style="font-size:1.05rem; color:#14532D; line-height:1.6; font-weight:500;">
                        {xai['layman_summary']}
                    </p>
                    <hr style="border:0.5px solid #86EFAC; margin:16px 0;">
                    <h4 style="color:#166534;">📋 Healthy Traffic Markers Confirmed:</h4>
                """, unsafe_allow_html=True)
                for r in xai["reasons"]:
                    st.markdown(f"- {r}")
                st.markdown(f"""
                    <div style="background-color:#DCFCE7; padding:12px; border-radius:8px; margin-top:14px; color:#166534; font-weight:600;">
                        🛡️ {xai['recommended_action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Risk Factor Table
            st.markdown("##### 📊 Telemetry Safety Indicators:")
            rf_df = pd.DataFrame(xai["risk_factors"])
            st.dataframe(rf_df, use_container_width=True)

    # --------------------------------------------------------------------------
    # DEMO TAB 2: BATCH CSV SCANNER
    # --------------------------------------------------------------------------
    with demo_tab2:
        st.subheader("📁 Bulk Network Traffic Scanner (CSV)")
        st.write("Scan an entire packet flow log containing hundreds or thousands of connections all at once.")

        c_up, c_demo = st.columns([1.2, 0.8])
        with c_demo:
            load_demo_batch = st.button("🧪 1-Click Load 1,000-Flow Demo Dataset", use_container_width=True)
        with c_up:
            up_file = st.file_uploader("Upload Traffic Log CSV:", type=["csv"])

        batch_input_df = None
        source_label = ""

        if up_file is not None:
            batch_input_df = pd.read_csv(up_file, low_memory=False)
            source_label = up_file.name
        elif load_demo_batch:
            demo_csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "demo_batch_sample.csv")
            if os.path.exists(demo_csv_path):
                batch_input_df = pd.read_csv(demo_csv_path)
                source_label = "demo_batch_sample.csv (500 Normal + 500 Attack Flows)"

        if batch_input_df is not None:
            st.success(f"Loaded `{source_label}` with **{len(batch_input_df):,} records**.")
            st.dataframe(batch_input_df.head(5), use_container_width=True)

            if st.button("⚡ Scan All Connections with Spark AI", type="primary"):
                with st.spinner(f"Analyzing {len(batch_input_df):,} connections in parallel with Spark MLlib..."):
                    res_df, summary = predict_batch_flows(batch_input_df, model_name="Random Forest")

                st.markdown("---")
                st.subheader("📊 Bulk Analysis Results")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Scanned", f"{summary['total_analyzed']:,}")
                with m2:
                    st.metric("Safe Traffic (BENIGN)", f"{summary['benign_count']:,}", f"{summary['benign_pct']}%")
                with m3:
                    st.metric("Hacker Probes (PortScan)", f"{summary['portscan_count']:,}", f"-{summary['portscan_pct']}%", delta_color="inverse")
                with m4:
                    st.metric("Detection Engine", "PySpark Random Forest")

                chart_df = pd.DataFrame({
                    "Category": ["Safe Traffic (BENIGN)", "Malicious Probes (PortScan)"],
                    "Count": [summary["benign_count"], summary["portscan_count"]]
                })
                st.bar_chart(chart_df.set_index("Category"), color="#0F766E")

                st.markdown("##### 📋 Inspection Table:")
                disp_cols = ["Predicted_Class", "Threat_Score (%)", "Confidence (%)", "destination_port", "flow_duration", "total_fwd_packets"]
                st.dataframe(res_df[disp_cols].head(500), use_container_width=True)

                csv_data = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Full Classified Report (CSV)",
                    data=csv_data,
                    file_name="spark_intrusion_detection_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==============================================================================
# PORTAL 2: ADMIN & SECURITY ENGINEER PORTAL (DEEP TECHNICAL TELEMETRY)
# ==============================================================================
else:
    # Admin Hero Banner
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);">
        <div class="hero-title">🛡️ Security Operations Center (SOC) Admin Portal</div>
        <div class="hero-subtitle">Distributed Big Data Telemetry, MLlib Benchmarks & Cluster Architecture</div>
        <span class="role-badge-admin">🛡️ Administrator / Engineering Mode Active</span>
    </div>
    """, unsafe_allow_html=True)

    admin_tabs = st.tabs([
        "📊 Cluster & Dataset Telemetry",
        "📈 Deep Model Benchmark",
        "🎯 Confusion Matrix Inspector",
        "🛠️ Pipeline Architecture"
    ])

    # ADMIN TAB 1: CLUSTER & DATASET
    with admin_tabs[0]:
        st.subheader("🖥️ Apache Spark Distributed Cluster Specs")
        ak1, ak2, ak3, ak4 = st.columns(4)
        with ak1:
            st.metric("Spark Runtime", "v3.5.9 (Local Multi-Core)")
        with ak2:
            st.metric("JVM Architecture", "OpenJDK 17 LTS (Hotspot)")
        with ak3:
            st.metric("Driver / Exec Memory", "4 GB Dedicated")
        with ak4:
            st.metric("Shuffle Partitions", "8 Partitions")

        st.markdown("---")
        st.subheader("🧹 Big Data Preprocessing & Hygiene Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Raw Rows", f"{prep_meta.get('raw_rows', 286467):,}")
        with c2:
            st.metric("Duplicates Dropped", f"{prep_meta.get('duplicates_removed', 72353):,}", "-25.2%")
        with c3:
            st.metric("Inf / NaN Rows Dropped", f"{prep_meta.get('missing_and_inf_rows_removed', 337):,}")
        with c4:
            st.metric("Cleaned Usable Records", f"{prep_meta.get('cleaned_rows', 213777):,}")

        col_prune, col_ratio = st.columns([1.1, 0.9])
        with col_prune:
            st.markdown("##### ✂️ Zero-Variance Feature Pruning (10 Columns Removed):")
            st.caption("Features having `nunique <= 1` across all 286k records were purged to eliminate zero-variance division anomalies:")
            st.code(", ".join(prep_meta.get("constant_columns_dropped", [])))
        with col_ratio:
            st.markdown("##### ⚖️ Balanced Class Ratio (Post-Hygiene):")
            st.write("- **BENIGN (0):** 123,083 (57.58%)")
            st.write("- **PortScan (1):** 90,694 (42.42%)")

    # ADMIN TAB 2: DEEP MODEL BENCHMARK
    with admin_tabs[1]:
        st.subheader("📈 PySpark MLlib Benchmark: 42,582 Held-Out Test Records")
        if model_results:
            b_rows = []
            for m in model_results:
                cm = m["confusion_matrix"]
                b_rows.append({
                    "Model": m["model_name"],
                    "Accuracy (%)": f"{m['accuracy']*100:.2f}%",
                    "Precision (%)": f"{m['precision']*100:.2f}%",
                    "Recall (%)": f"{m['recall']*100:.2f}%",
                    "F1-Score (%)": f"{m['f1_score']*100:.2f}%",
                    "ROC-AUC": f"{m['roc_auc']:.4f}",
                    "Train Latency": f"{m['train_time_sec']}s",
                    "Test Latency": f"{m['inference_time_sec']}s",
                    "TN (Benign)": f"{cm['tn']:,}",
                    "FP (False Alarm)": f"{cm['fp']:,}",
                    "FN (Missed Attack)": f"{cm['fn']:,}",
                    "TP (Caught Attack)": f"{cm['tp']:,}"
                })
            st.dataframe(pd.DataFrame(b_rows), use_container_width=True)

        comp_bar = os.path.join(PROJECT_ROOT, "results", "figures", "model_comparison_bar.png")
        if os.path.exists(comp_bar):
            st.image(comp_bar, caption="Spark MLlib Model Benchmark: Accuracy, Precision, Recall & F1", use_container_width=True)

    # ADMIN TAB 3: CONFUSION MATRIX INSPECTOR
    with admin_tabs[2]:
        st.subheader("🎯 Confusion Matrix Heatmap Inspector")
        inspect_pick = st.selectbox(
            "Select Model:",
            ["Random Forest", "Gradient-Boosted Trees", "Logistic Regression"]
        )
        tag = inspect_pick.lower().replace(" ", "_").replace("-", "_")
        cm_path = os.path.join(PROJECT_ROOT, "results", "figures", f"{tag}_cm.png")
        if not os.path.exists(cm_path) and "boost" in tag:
            cm_path = os.path.join(PROJECT_ROOT, "results", "figures", "gradient-boosted_trees_cm.png")

        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix: {inspect_pick} (42,582 Test Samples)", width=550)

    # ADMIN TAB 4: PIPELINE ARCHITECTURE
    with admin_tabs[3]:
        st.subheader("🛠️ Distributed Spark Pipeline Specs")
        st.markdown(
            """
            ```text
            Raw CSV (286,467 records)
                │
                ▼
            cleaner.py ──> Drop Duplicates (72k) ──> Replace Inf/NaN ──> Drop 10 Constant Columns
                │
                ▼
            PySpark DataFrame (213,777 records)
                │
                ├── Train Partition (80%: 171,195 rows, seed=42)
                └── Test Partition  (20%: 42,582 rows, seed=42)
                │
                ▼
            feature_pipeline.py
                ├── VectorAssembler(inputCols=68 features, outputCol="raw_features")
                └── StandardScaler(inputCol="raw_features", outputCol="features", withStd=True)
                │
                ▼
            PySpark MLlib Classifiers
                ├── RandomForestClassifier(numTrees=30, maxDepth=10) -> 99.99% Acc, 0 False Positives
                ├── GBTClassifier(maxIter=30, maxDepth=5)            -> 99.96% Acc
                └── LogisticRegression(maxIter=50, regParam=0.01)     -> 99.00% Acc
            ```
            """
        )
        st.info("Full project report documentation available in `results/FINAL_PROJECT_REPORT.md`.")
