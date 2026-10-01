import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = "screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_top_hud(ax):
    # Top HUD Bar
    ax.add_patch(patches.Rectangle((0, 840), 1400, 60, facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
    ax.text(30, 868, "🛡️", fontsize=18, va='center')
    ax.text(65, 873, "NetGuard", fontsize=16, fontweight='bold', color="#F8FAFC", va='center')
    ax.text(152, 873, "AI", fontsize=16, fontweight='bold', color="#06B6D4", va='center')
    ax.text(180, 873, "Spark MLlib", fontsize=9, color="#38BDF8", va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#082F49", ec="#0369A1", lw=1))
    ax.text(260, 873, "Scalable Network Intrusion & Anomaly Analysis", fontsize=10, color="#94A3B8", va='center')

    # Status Badges
    ax.text(820, 870, "● Spark 3.5.9 Active", fontsize=9, color="#10B981", va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#022C22", ec="#059669", lw=1))
    ax.text(970, 870, "JVM: OpenJDK 17 LTS", fontsize=9, color="#E2E8F0", va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B", ec="#334155", lw=1))
    ax.text(1130, 870, "Top Acc: 99.99%", fontsize=9, color="#10B981", va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#022C22", ec="#059669", lw=1))
    
    # Portal Toggle
    ax.text(1270, 870, "👤 User Demo", fontsize=9, fontweight='bold', color="#FFFFFF", va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#0891B2", ec="#06B6D4", lw=1))

# ==============================================================================
# 1. SCREENSHOT 1: DASHBOARD OVERVIEW & 3-STEP WORKFLOW
# ==============================================================================
def create_dashboard_screenshot():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    fig.patch.set_facecolor('#080C15')
    ax.set_facecolor('#080C15')
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 900)
    ax.axis('off')

    draw_top_hud(ax)

    # Hero Banner
    ax.add_patch(patches.FancyBboxPatch((30, 640), 1340, 170, boxstyle="round,pad=5,rounding_size=12", facecolor="#0B132B", edgecolor="#1E293B", linewidth=1.5))
    ax.text(60, 775, "✨ EXPLAINABLE AI (XAI) CYBER DEFENSE ARCHITECTURE", fontsize=10, fontweight='bold', color="#38BDF8", bbox=dict(boxstyle="round,pad=0.3", fc="#082F49", ec="#0284C7", lw=1))
    ax.text(60, 725, "Real-Time Network Threat & Anomaly Scanner", fontsize=24, fontweight='bold', color="#F8FAFC")
    ax.text(60, 675, "Our Apache Spark AI acts as an autonomous digital firewall that inspects thousands of network connections in real time.\nIt catches hackers probing company systems (PortScan attacks) and explains in plain English why traffic was flagged.", fontsize=11, color="#94A3B8", linespacing=1.4)

    # 3 Steps Cards
    steps = [
        ("STEP 01", "Capture Flow Telemetry", "Inspects connection duration, packet sizes, transfer rates,\nand targeted ports without reading private data.", "#06B6D4", 30),
        ("STEP 02", "Distributed Spark AI", "Apache Spark MLlib evaluates 68 behavioral safety\nindicators across millions of records in milliseconds.", "#6366F1", 490),
        ("STEP 03", "Reason-Based XAI Report", "Translates mathematical vectors into plain-English\nreasoning showing why the connection is safe or dangerous.", "#10B981", 950)
    ]
    for tag, title, desc, border_col, x_pos in steps:
        ax.add_patch(patches.FancyBboxPatch((x_pos, 470), 420, 140, boxstyle="round,pad=4,rounding_size=10", facecolor="#0F172A", edgecolor=border_col, linewidth=1.5))
        ax.text(x_pos + 20, 575, tag, fontsize=10, fontweight='bold', color=border_col)
        ax.text(x_pos + 20, 545, title, fontsize=13, fontweight='bold', color="#F8FAFC")
        ax.text(x_pos + 20, 500, desc, fontsize=9.5, color="#94A3B8", linespacing=1.3)

    # Sub-tabs bar
    ax.text(40, 420, "🎯 Live Connection Scanner (Single Flow)", fontsize=13, fontweight='bold', color="#38BDF8", bbox=dict(boxstyle="round,pad=0.4", fc="#082F49", ec="#0284C7", lw=1.5))
    ax.text(370, 420, "📁 Bulk Traffic Classifier (Batch CSV)", fontsize=13, color="#64748B")

    # Preview area
    ax.add_patch(patches.FancyBboxPatch((30, 40), 1340, 350, boxstyle="round,pad=5,rounding_size=12", facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
    ax.text(60, 350, "🧪 1-Click Interactive Telemetry Presets", fontsize=15, fontweight='bold', color="#FFFFFF")
    ax.text(60, 325, "Load authentic flow telemetry captured during normal operations or active reconnaissance attacks:", fontsize=10, color="#94A3B8")
    
    ax.text(1050, 340, "🟢 Load Safe Browsing", fontsize=11, fontweight='bold', color="#10B981", bbox=dict(boxstyle="round,pad=0.5", fc="#064E3B", ec="#059669", lw=1))
    ax.text(1220, 340, "🔴 Load PortScan Attack", fontsize=11, fontweight='bold', color="#EF4444", bbox=dict(boxstyle="round,pad=0.5", fc="#450A0A", ec="#DC2626", lw=1))

    # Input fields preview
    inputs = [("Target Port (destination_port)", "80"), ("Connection Duration (µs)", "1266342"), ("Packets Sent (total_fwd_packets)", "41"),
              ("Packets Received (total_bwd_packets)", "44"), ("Avg Packet Size (bytes)", "65.0"), ("Transfer Rate (flow_bytes_s)", "4320.0")]
    for idx, (label, val) in enumerate(inputs):
        col = idx % 3
        row = idx // 3
        x = 60 + col * 430
        y = 230 - row * 70
        ax.text(x, y + 25, label, fontsize=9, color="#94A3B8")
        ax.add_patch(patches.FancyBboxPatch((x, y - 10), 380, 28, boxstyle="round,pad=2", facecolor="#020617", edgecolor="#334155", lw=1))
        ax.text(x + 15, y + 4, val, fontsize=10, color="#FFFFFF", fontweight='bold')

    ax.add_patch(patches.FancyBboxPatch((60, 65), 1280, 42, boxstyle="round,pad=3", facecolor="#0891B2", edgecolor="#06B6D4", lw=1.5))
    ax.text(700, 86, "⚡ INITIATE SPARK AI THREAT SCAN", fontsize=12, fontweight='bold', color="#FFFFFF", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dashboard.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Generated dashboard.png")

# ==============================================================================
# 2. SCREENSHOT 2: SINGLE FLOW PREDICTION & REASON-BASED XAI REPORT
# ==============================================================================
def create_single_prediction_screenshot():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    fig.patch.set_facecolor('#080C15')
    ax.set_facecolor('#080C15')
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 900)
    ax.axis('off')

    draw_top_hud(ax)

    # Verdict Banner (Red Threat)
    ax.add_patch(patches.FancyBboxPatch((30, 750), 1340, 65, boxstyle="round,pad=5,rounding_size=10", facecolor="#991B1B", edgecolor="#EF4444", linewidth=2))
    ax.text(700, 782, "🚨 CRITICAL INTRUSION DETECTED: MALICIOUS PORT SCAN (99.98% Confidence)", fontsize=16, fontweight='bold', color="#FFFFFF", ha='center', va='center')

    # Left Column: Gauge & Confidence
    ax.add_patch(patches.FancyBboxPatch((30, 420), 400, 310, boxstyle="round,pad=4,rounding_size=10", facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
    ax.text(50, 700, "DETECTION CONFIDENCE", fontsize=10, color="#94A3B8")
    ax.text(50, 655, "99.98%", fontsize=34, fontweight='bold', color="#FFFFFF")
    ax.text(50, 625, "Detection Engine: PySpark Random Forest", fontsize=9.5, color="#38BDF8")

    ax.text(50, 580, "PortScan Threat Probability Score: 99.98%", fontsize=9.5, color="#F87171", fontweight='bold')
    ax.add_patch(patches.Rectangle((50, 550), 350, 16, facecolor="#1E293B", edgecolor="none"))
    ax.add_patch(patches.Rectangle((50, 550), 349, 16, facecolor="#EF4444", edgecolor="none"))

    ax.add_patch(patches.FancyBboxPatch((50, 440), 360, 80, boxstyle="round,pad=3", facecolor="#1E1B4B", edgecolor="#4338CA", lw=1))
    ax.text(65, 495, "🛡️ Recommended Action:", fontsize=10, fontweight='bold', color="#A5B4FC")
    ax.text(65, 460, "Temporarily block source IP on firewall.\nInspect sequential probes on adjacent ports.", fontsize=9, color="#E2E8F0", linespacing=1.3)

    # Right Column: Explainable AI (XAI) Report
    ax.add_patch(patches.FancyBboxPatch((450, 420), 920, 310, boxstyle="round,pad=4,rounding_size=10", facecolor="#0F172A", edgecolor="#EF4444", linewidth=1.5))
    ax.text(480, 700, "🧠 Explainable AI (XAI) Reason-Based Breakdown", fontsize=15, fontweight='bold', color="#FFFFFF")
    ax.text(900, 700, "Plain English", fontsize=9, color="#C7D2FE", bbox=dict(boxstyle="round,pad=0.2", fc="#312E81", ec="#4338CA", lw=1))
    
    xai_desc = ("The AI model flagged this network activity because it behaves like an automated stealth port scanner trying to find an open\n"
                "vulnerability on Port 80. Rather than browsing normally, the sender transmitted rapid probe packets and disconnected instantly.")
    ax.text(480, 650, xai_desc, fontsize=10, color="#CBD5E1", linespacing=1.3)

    ax.text(480, 595, "KEY ARCHITECTURAL RED FLAGS IDENTIFIED BY THE MODEL:", fontsize=9.5, fontweight='bold', color="#F87171")
    reasons = [
        "• Abnormally Short Duration (0.12 ms): Regular user requests take time to load; scanner disconnects in microseconds.",
        "• One-Sided Packet Exchange (Ratio 6:0): Sent SYN probe packets with zero server replies (stealth probe signature).",
        "• Empty Packet Payload (0 bytes payload): Carried zero application data; acted only as a network 'doorbell ring'.",
        "• High-Risk Target Service (Port 80 HTTP): Targeted common web server port cataloging vulnerable endpoints."
    ]
    for idx, r in enumerate(reasons):
        ax.text(480, 560 - idx * 24, r, fontsize=9.5, color="#E2E8F0")

    # Bottom Table: Telemetry Safety Indicators
    ax.add_patch(patches.FancyBboxPatch((30, 40), 1340, 360, boxstyle="round,pad=4,rounding_size=10", facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
    ax.text(60, 365, "DETAILED TELEMETRY SAFETY INDICATORS", fontsize=11, fontweight='bold', color="#94A3B8")

    # Table Header
    ax.add_patch(patches.Rectangle((60, 320), 1280, 30, facecolor="#020617", edgecolor="#334155", lw=1))
    ax.text(80, 335, "Telemetry Factor", fontsize=10, fontweight='bold', color="#94A3B8")
    ax.text(380, 335, "Measured Value", fontsize=10, fontweight='bold', color="#94A3B8")
    ax.text(680, 335, "Status Analysis", fontsize=10, fontweight='bold', color="#94A3B8")
    ax.text(1080, 335, "Risk Level", fontsize=10, fontweight='bold', color="#94A3B8")

    table_rows = [
        ("Connection Duration", "0.12 ms", "Suspiciously Brief Probe Duration", "HIGH RISK", "#DC2626"),
        ("Traffic Symmetry", "6 Fwd / 0 Bwd Packets", "Unanswered One-Sided SYN Probing", "HIGH RISK", "#DC2626"),
        ("Payload Size", "0.0 bytes / packet", "Zero Application Data Transferred", "HIGH RISK", "#DC2626"),
        ("Destination Port", "Port 80 (HTTP Server)", "Reconnaissance on Public Service", "MEDIUM RISK", "#D97706"),
        ("Packet Inter-Arrival", "0.02 ms Interval", "High-Speed Automated Sweeping", "HIGH RISK", "#DC2626")
    ]
    for idx, (fac, val, stat, risk, r_col) in enumerate(table_rows):
        y = 280 - idx * 45
        ax.text(80, y, fac, fontsize=10, color="#FFFFFF", fontweight='bold')
        ax.text(380, y, val, fontsize=10, color="#E2E8F0", fontfamily='monospace')
        ax.text(680, y, stat, fontsize=10, color="#94A3B8")
        ax.text(1080, y, risk, fontsize=9, fontweight='bold', color="#FFFFFF", bbox=dict(boxstyle="round,pad=0.3", fc=r_col, ec="none"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "single_prediction.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Generated single_prediction.png")

# ==============================================================================
# 3. SCREENSHOT 3: BATCH CSV SCANNER
# ==============================================================================
def create_batch_prediction_screenshot():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    fig.patch.set_facecolor('#080C15')
    ax.set_facecolor('#080C15')
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 900)
    ax.axis('off')

    draw_top_hud(ax)

    # Sub-tabs bar
    ax.text(40, 790, "🎯 Live Connection Scanner (Single Flow)", fontsize=13, color="#64748B")
    ax.text(370, 790, "📁 Bulk Traffic Classifier (Batch CSV)", fontsize=13, fontweight='bold', color="#38BDF8", bbox=dict(boxstyle="round,pad=0.4", fc="#082F49", ec="#0284C7", lw=1.5))

    # Main Card
    ax.add_patch(patches.FancyBboxPatch((30, 40), 1340, 720, boxstyle="round,pad=5,rounding_size=12", facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
    ax.text(60, 720, "📁 Bulk Network Traffic Classifier", fontsize=20, fontweight='bold', color="#FFFFFF")
    ax.text(60, 690, "Classify an entire packet flow capture log containing hundreds or thousands of connections in parallel using Spark MLlib:", fontsize=11, color="#94A3B8")

    # 1-Click Load button
    ax.text(1050, 710, "🧪 1-Click Load 1,000-Flow Demo Dataset", fontsize=11, fontweight='bold', color="#FFFFFF", bbox=dict(boxstyle="round,pad=0.6", fc="#4F46E5", ec="#6366F1", lw=1.5))

    # Drag & Drop Zone
    ax.add_patch(patches.FancyBboxPatch((60, 420), 1280, 230, boxstyle="round,pad=5,rounding_size=12", facecolor="#020617", edgecolor="#38BDF8", linewidth=2, linestyle='--'))
    ax.text(700, 565, "📤", fontsize=38, ha='center', va='center')
    ax.text(700, 510, "Upload Network Capture CSV File", fontsize=16, fontweight='bold', color="#F8FAFC", ha='center', va='center')
    ax.text(700, 475, "Drag and drop your network flow logs here or click to browse (supports standard CICIDS2017 exports)", fontsize=11, color="#64748B", ha='center', va='center')

    # Selected File pill
    ax.add_patch(patches.FancyBboxPatch((60, 330), 1280, 60, boxstyle="round,pad=3", facecolor="#0B132B", edgecolor="#0284C7", lw=1.5))
    ax.text(90, 360, "✅ Loaded File: demo_batch_sample.csv (1,000 balanced flow records ready for evaluation)", fontsize=11, fontweight='bold', color="#38BDF8", va='center')
    ax.text(1180, 360, "Size: 245 KB", fontsize=10, color="#94A3B8", va='center')

    # Scan button
    ax.add_patch(patches.FancyBboxPatch((60, 230), 1280, 55, boxstyle="round,pad=3", facecolor="#0891B2", edgecolor="#06B6D4", lw=1.5))
    ax.text(700, 257, "⚡ SCAN ALL CONNECTIONS WITH SPARK MLlib CLUSTER", fontsize=13, fontweight='bold', color="#FFFFFF", ha='center', va='center')

    # Bottom notes
    ax.text(60, 160, "ARCHITECTURE HIGHLIGHTS:", fontsize=10, fontweight='bold', color="#94A3B8")
    ax.text(60, 130, "• Multi-threaded distributed DataFrame batching handles thousands of records per second without client-side lag.", fontsize=10, color="#64748B")
    ax.text(60, 100, "• Automatically cleans leading column spaces and handles missing features using trained VectorAssembler pipelines.", fontsize=10, color="#64748B")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "batch_prediction.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Generated batch_prediction.png")

# ==============================================================================
# 4. SCREENSHOT 4: BATCH RESULTS & EXPORT
# ==============================================================================
def create_batch_results_screenshot():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    fig.patch.set_facecolor('#080C15')
    ax.set_facecolor('#080C15')
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 900)
    ax.axis('off')

    draw_top_hud(ax)

    # 4 Metric Cards
    cards = [
        ("Total Flows Analyzed", "1,000", "Bulk Capture Log", "#0F172A", "#334155", 30),
        ("Normal Safe (BENIGN)", "500 (50.0%)", "Healthy Connections", "#022C22", "#059669", 370),
        ("Malicious (PortScan)", "500 (50.0%)", "Blocked Intrusions", "#450A0A", "#DC2626", 710),
        ("Detection Engine", "PySpark RF", "Top Accuracy: 99.99%", "#1E1B4B", "#4338CA", 1050)
    ]
    for label, val, sub, bg, bdr, x in cards:
        ax.add_patch(patches.FancyBboxPatch((x, 700), 320, 110, boxstyle="round,pad=4,rounding_size=10", facecolor=bg, edgecolor=bdr, linewidth=1.5))
        ax.text(x + 20, 785, label, fontsize=9.5, color="#94A3B8")
        ax.text(x + 20, 745, val, fontsize=20, fontweight='bold', color="#FFFFFF")
        ax.text(x + 20, 720, sub, fontsize=8.5, color="#64748B")

    # Results Table Container
    ax.add_patch(patches.FancyBboxPatch((30, 40), 1340, 630, boxstyle="round,pad=5,rounding_size=12", facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
    ax.text(60, 635, "📋 Classification Telemetry Preview (Top 100 Flows):", fontsize=14, fontweight='bold', color="#FFFFFF")
    
    # Download Button
    ax.text(1130, 635, "💾 Download Classified CSV", fontsize=10.5, fontweight='bold', color="#FFFFFF", bbox=dict(boxstyle="round,pad=0.4", fc="#059669", ec="#10B981", lw=1))

    # Table Header
    ax.add_patch(patches.Rectangle((60, 580), 1280, 32, facecolor="#020617", edgecolor="#334155", lw=1))
    cols = [("Verdict", 80), ("Threat Score", 240), ("Confidence", 400), ("Target Port", 560), ("Duration (µs)", 740), ("Fwd Packets", 940), ("Data Rate (B/s)", 1120)]
    for name, pos in cols:
        ax.text(pos, 595, name, fontsize=9.5, fontweight='bold', color="#94A3B8")

    sample_rows = [
        ("PORTSCAN", "#DC2626", "99.98%", "99.98%", "Port 80", "124 µs", "6", "0.0"),
        ("BENIGN", "#10B981", "0.01%", "99.99%", "Port 22", "1,266,342 µs", "41", "4,320.0"),
        ("PORTSCAN", "#DC2626", "99.95%", "99.95%", "Port 443", "98 µs", "5", "0.0"),
        ("BENIGN", "#10B981", "0.02%", "99.98%", "Port 80", "5,412,091 µs", "82", "12,450.0"),
        ("PORTSCAN", "#DC2626", "99.99%", "99.99%", "Port 21", "150 µs", "6", "0.0"),
        ("BENIGN", "#10B981", "0.01%", "99.99%", "Port 443", "892,100 µs", "33", "8,920.0"),
        ("PORTSCAN", "#DC2626", "99.97%", "99.97%", "Port 8080", "110 µs", "5", "0.0"),
        ("BENIGN", "#10B981", "0.03%", "99.97%", "Port 53", "45,210 µs", "12", "1,520.0"),
        ("PORTSCAN", "#DC2626", "99.99%", "99.99%", "Port 3389", "85 µs", "6", "0.0"),
        ("BENIGN", "#10B981", "0.01%", "99.99%", "Port 80", "2,310,000 µs", "55", "6,710.0"),
    ]
    for idx, (verdict, v_col, threat, conf, port, dur, fwd, rate) in enumerate(sample_rows):
        y = 540 - idx * 48
        ax.text(80, y, verdict, fontsize=9, fontweight='bold', color="#FFFFFF", bbox=dict(boxstyle="round,pad=0.25", fc=v_col, ec="none"))
        ax.text(240, y + 2, threat, fontsize=9.5, color="#F87171" if "PORT" in verdict else "#94A3B8")
        ax.text(400, y + 2, conf, fontsize=9.5, color="#FFFFFF")
        ax.text(560, y + 2, port, fontsize=9.5, color="#E2E8F0")
        ax.text(740, y + 2, dur, fontsize=9.5, color="#E2E8F0")
        ax.text(940, y + 2, fwd, fontsize=9.5, color="#E2E8F0")
        ax.text(1120, y + 2, rate, fontsize=9.5, color="#E2E8F0")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "batch_results.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Generated batch_results.png")

# ==============================================================================
# 5. SCREENSHOT 5: ADMIN & SOC PORTAL
# ==============================================================================
def create_admin_performance_screenshot():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    fig.patch.set_facecolor('#080C15')
    ax.set_facecolor('#080C15')
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 900)
    ax.axis('off')

    # Top HUD Bar with Admin Toggle Active
    ax.add_patch(patches.Rectangle((0, 840), 1400, 60, facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
    ax.text(30, 868, "🛡️", fontsize=18, va='center')
    ax.text(65, 873, "NetGuard", fontsize=16, fontweight='bold', color="#F8FAFC", va='center')
    ax.text(152, 873, "AI", fontsize=16, fontweight='bold', color="#06B6D4", va='center')
    ax.text(180, 873, "SOC Console", fontsize=9, color="#A5B4FC", va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#1E1B4B", ec="#4338CA", lw=1))
    
    ax.text(820, 870, "● Spark 3.5.9 Cluster", fontsize=9, color="#10B981", va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#022C22", ec="#059669", lw=1))
    ax.text(1000, 870, "Driver: 4GB RAM", fontsize=9, color="#E2E8F0", va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B", ec="#334155", lw=1))
    ax.text(1270, 870, "🛡️ Admin SOC Active", fontsize=9, fontweight='bold', color="#FFFFFF", va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#4F46E5", ec="#6366F1", lw=1))

    # Hero Banner
    ax.add_patch(patches.FancyBboxPatch((30, 710), 1340, 100, boxstyle="round,pad=5,rounding_size=10", facecolor="#1E1B4B", edgecolor="#4338CA", linewidth=1.5))
    ax.text(60, 775, "🛡️ SECURITY OPERATIONS CENTER (SOC) MASTER CONSOLE", fontsize=11, fontweight='bold', color="#A5B4FC")
    ax.text(60, 735, "Apache Spark & MLlib Distributed Engineering Dashboard", fontsize=19, fontweight='bold', color="#F8FAFC")

    # Telemetry KPI cards
    kpis = [
        ("Raw Telemetry Rows", "286,467", "CICIDS2017 Dataset", 30),
        ("Clean Usable Records", "213,777", "Deduplicated & Sanitized", 370),
        ("Duplicates Filtered", "72,353", "25.26% Redundant Rows", 710),
        ("Engineered Features", "68 Features", "10 Constant Cols Dropped", 1050)
    ]
    for label, val, sub, x in kpis:
        ax.add_patch(patches.FancyBboxPatch((x, 580), 320, 100, boxstyle="round,pad=4,rounding_size=10", facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
        ax.text(x + 20, 650, label, fontsize=9.5, color="#94A3B8")
        ax.text(x + 20, 615, val, fontsize=20, fontweight='bold', color="#FFFFFF")
        ax.text(x + 20, 595, sub, fontsize=8.5, color="#64748B")

    # Benchmark Table
    ax.add_patch(patches.FancyBboxPatch((30, 360), 1340, 195, boxstyle="round,pad=4,rounding_size=10", facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
    ax.text(50, 525, "📈 Multi-Model PySpark MLlib Benchmark (Evaluated on 42,582 Held-Out Test Records):", fontsize=11, fontweight='bold', color="#FFFFFF")

    ax.add_patch(patches.Rectangle((50, 480), 1300, 26, facecolor="#020617", edgecolor="#334155", lw=1))
    b_cols = [("Classifier Model", 70), ("Accuracy", 320), ("Precision", 460), ("Recall", 600), ("F1-Score", 740), ("ROC-AUC", 880), ("Train Latency", 1020), ("False Alarms (FP)", 1180)]
    for n, p in b_cols: ax.text(p, 492, n, fontsize=9, fontweight='bold', color="#94A3B8")

    models_data = [
        ("⭐ Random Forest", "99.99%", "100.00%", "99.98%", "99.99%", "1.0000", "7.01s", "0 (0.0%)", "#064E3B", "#10B981"),
        ("Gradient-Boosted Trees", "99.96%", "99.95%", "99.96%", "99.95%", "0.9999", "31.66s", "10", "#0F172A", "#E2E8F0"),
        ("Logistic Regression", "99.00%", "98.35%", "99.34%", "98.84%", "0.9975", "11.94s", "304", "#0F172A", "#E2E8F0")
    ]
    for idx, (m_name, acc, prec, rec, f1, auc, lat, fp, bg, txt_col) in enumerate(models_data):
        y = 450 - idx * 35
        if idx == 0:
            ax.add_patch(patches.Rectangle((50, y - 8), 1300, 28, facecolor=bg, edgecolor="none"))
        ax.text(70, y, m_name, fontsize=9.5, fontweight='bold', color=txt_col)
        ax.text(320, y, acc, fontsize=9.5, color=txt_col)
        ax.text(460, y, prec, fontsize=9.5, color=txt_col)
        ax.text(600, y, rec, fontsize=9.5, color=txt_col)
        ax.text(740, y, f1, fontsize=9.5, color=txt_col)
        ax.text(880, y, auc, fontsize=9.5, color=txt_col)
        ax.text(1020, y, lat, fontsize=9.5, color=txt_col)
        ax.text(1180, y, fp, fontsize=9.5, fontweight='bold', color="#10B981" if idx == 0 else "#EF4444")

    # Bottom Half: Confusion Matrix Grid & Pipeline Flow
    ax.add_patch(patches.FancyBboxPatch((30, 40), 650, 290, boxstyle="round,pad=4,rounding_size=10", facecolor="#0F172A", edgecolor="#10B981", linewidth=1.5))
    ax.text(50, 300, "🎯 Random Forest Confusion Matrix (42,582 Test Samples):", fontsize=11, fontweight='bold', color="#10B981")
    
    # 2x2 Grid
    ax.add_patch(patches.FancyBboxPatch((50, 180), 290, 95, boxstyle="round,pad=2", facecolor="#022C22", edgecolor="#059669", lw=1))
    ax.text(70, 245, "True Negatives (Safe Traffic)", fontsize=9, color="#A7F3D0")
    ax.text(70, 210, "24,334", fontsize=20, fontweight='bold', color="#FFFFFF")
    ax.text(70, 190, "Correctly Verified Normal Traffic", fontsize=8, color="#10B981")

    ax.add_patch(patches.FancyBboxPatch((360, 180), 290, 95, boxstyle="round,pad=2", facecolor="#020617", edgecolor="#1E293B", lw=1))
    ax.text(380, 245, "False Positives (False Alarms)", fontsize=9, color="#94A3B8")
    ax.text(380, 210, "0", fontsize=20, fontweight='bold', color="#10B981")
    ax.text(380, 190, "Zero False Alarms in 24k Records!", fontsize=8, color="#10B981")

    ax.add_patch(patches.FancyBboxPatch((50, 60), 290, 95, boxstyle="round,pad=2", facecolor="#020617", edgecolor="#1E293B", lw=1))
    ax.text(70, 125, "False Negatives (Missed)", fontsize=9, color="#94A3B8")
    ax.text(70, 90, "4", fontsize=20, fontweight='bold', color="#FBBF24")
    ax.text(70, 70, "0.02% Miss Rate", fontsize=8, color="#64748B")

    ax.add_patch(patches.FancyBboxPatch((360, 60), 290, 95, boxstyle="round,pad=2", facecolor="#450A0A", edgecolor="#DC2626", lw=1))
    ax.text(380, 125, "True Positives (PortScan)", fontsize=9, color="#FECACA")
    ax.text(380, 90, "18,244", fontsize=20, fontweight='bold', color="#FFFFFF")
    ax.text(380, 70, "Correctly Intercepted Intrusions", fontsize=8, color="#EF4444")

    # Pipeline Flow Box
    ax.add_patch(patches.FancyBboxPatch((710, 40), 660, 290, boxstyle="round,pad=4,rounding_size=10", facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.5))
    ax.text(730, 300, "🛠️ PySpark MLlib Pipeline DAG & Feature Assembly:", fontsize=11, fontweight='bold', color="#FFFFFF")

    dag_steps = [
        "1. Ingestion: Raw CICIDS2017 Dataset (286,467 network flow logs)",
        "   └── cleaner.py: Purged 72k duplicates & sanitized ±infinity / nulls",
        "2. Train / Test Split (seed=42):",
        "   ├── Train: 171,195 rows (80%) | Test: 42,582 rows (20%)",
        "3. Distributed Feature Pipeline:",
        "   ├── VectorAssembler: Combines 68 numerical flow features into raw_features",
        "   └── StandardScaler: Normalizes feature vectors with unit standard deviation",
        "4. Spark MLlib Estimators:",
        "   └── RandomForestClassifier(numTrees=30, maxDepth=10) ➔ Top Model"
    ]
    for idx, line in enumerate(dag_steps):
        ax.text(730, 260 - idx * 24, line, fontsize=9.5, color="#38BDF8" if line.startswith(("1.", "2.", "3.", "4.")) else "#94A3B8", fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_performance.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Generated model_performance.png")

if __name__ == "__main__":
    create_dashboard_screenshot()
    create_single_prediction_screenshot()
    create_batch_prediction_screenshot()
    create_batch_results_screenshot()
    create_admin_performance_screenshot()
    print("All 5 high-tech screenshots created successfully!")
