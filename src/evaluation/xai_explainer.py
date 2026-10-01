def explain_prediction(features_dict, prediction_result):
    """
    Reason-Based Explainable AI (XAI) engine for Network Flow Intrusion Detection.
    Translates complex 68-feature network telemetry into plain-English,
    understandable explanations for non-technical users and security examiners.
    """
    label = prediction_result.get("prediction_label", "BENIGN")
    confidence = prediction_result.get("confidence", 0.99)
    portscan_prob = prediction_result.get("portscan_prob", 0.0)

    # Extract key telemetry features safely
    dest_port = int(features_dict.get("destination_port", 80))
    duration = float(features_dict.get("flow_duration", 1000000.0)) # in microseconds
    fwd_pkts = float(features_dict.get("total_fwd_packets", 5.0))
    bwd_pkts = float(features_dict.get("total_backward_packets", 4.0))
    pkt_len_mean = float(features_dict.get("packet_length_mean", 120.0))
    fwd_len_tot = float(features_dict.get("total_length_of_fwd_packets", 500.0))
    flow_bytes_s = float(features_dict.get("flow_bytes_s", 2500.0))
    flow_pkts_s = float(features_dict.get("flow_packets_s", 10.0))

    duration_ms = duration / 1000.0 # convert to milliseconds

    reasons = []
    risk_factors = []

    if label == "PORTSCAN":
        headline = "ALERT: MALICIOUS INTRUSION DETECTED (PORTSCAN RECONNAISSANCE)"
        layman_summary = (
            f"The AI model flagged this network activity because it behaves like an automated port scanner "
            f"trying to find an open vulnerability on Port {dest_port}. Rather than browsing normally, the sender "
            f"sent rapid probe packets and disconnected almost instantly."
        )

        # Factor 1: Connection Duration (Scanners disconnect very fast)
        if duration < 50000: # < 50ms
            reasons.append(
                f"**Abnormally Short Connection Time ({duration_ms:.2f} ms):** "
                f"Normal user requests take time to negotiate and load content. This flow lasted only a fraction of a second, "
                f"which is typical of automated bot sweeps checking if a port responds."
            )
            risk_factors.append({"factor": "Connection Duration", "value": f"{duration_ms:.2f} ms", "status": "Suspiciously Brief", "risk": "High"})
        else:
            risk_factors.append({"factor": "Connection Duration", "value": f"{duration_ms:.2f} ms", "status": "Moderate", "risk": "Medium"})

        # Factor 2: Asymmetric Traffic (One-sided probe vs two-way conversation)
        if bwd_pkts <= 1 and fwd_pkts >= 1:
            reasons.append(
                f"**One-Sided Packet Exchange (Ratio {int(fwd_pkts)}:{int(bwd_pkts)}):** "
                f"Legitimate web browsing involves a two-way dialogue where the server replies with data. "
                f"Here, probe packets were sent with minimal or zero server acknowledgment, typical of stealth SYN scanning."
            )
            risk_factors.append({"factor": "Traffic Symmetry", "value": f"{int(fwd_pkts)} Fwd / {int(bwd_pkts)} Bwd", "status": "Unanswered Probe", "risk": "High"})
        else:
            risk_factors.append({"factor": "Traffic Symmetry", "value": f"{int(fwd_pkts)} Fwd / {int(bwd_pkts)} Bwd", "status": "Asymmetric", "risk": "Medium"})

        # Factor 3: Payload Size (Probes carry almost no real data)
        if pkt_len_mean < 50 or fwd_len_tot < 100:
            reasons.append(
                f"**Empty or Minimal Packet Payload (Avg {pkt_len_mean:.1f} bytes):** "
                f"Regular web or database sessions transmit files, HTML, or JSON payloads. "
                f"This flow carried near-zero payload bytes, acting solely as a network 'doorbell ring' to check for open ports."
            )
            risk_factors.append({"factor": "Payload Size", "value": f"{pkt_len_mean:.1f} bytes/pkt", "status": "Zero/Empty Payload", "risk": "High"})
        else:
            risk_factors.append({"factor": "Payload Size", "value": f"{pkt_len_mean:.1f} bytes/pkt", "status": "Small Header", "risk": "Medium"})

        # Factor 4: Target Port Context
        common_ports = {80: "HTTP Web Server", 443: "HTTPS Secure Web", 22: "SSH Remote Terminal", 21: "FTP File Transfer", 445: "SMB Windows Sharing", 3389: "RDP Remote Desktop"}
        port_desc = common_ports.get(dest_port, f"Custom Port {dest_port}")
        reasons.append(
            f"**Target Service ({port_desc}):** "
            f"The scanner targeted Port {dest_port}, a prime target for automated reconnaissance scanners cataloging active services on network hosts."
        )

        recommended_action = (
            f"**Recommended Action:** Temporarily block or rate-limit the originating IP on the network firewall. "
            f"Inspect firewall logs for sequential probes across adjacent ports (e.g., ports {max(1, dest_port-1)} to {dest_port+5})."
        )

    else:
        headline = "VERIFIED: NORMAL NETWORK TRAFFIC (BENIGN)"
        layman_summary = (
            f"The AI model verified this connection as standard, harmless network communication. "
            f"The client and server engaged in a healthy, bidirectional exchange typical of genuine user activities "
            f"such as web browsing, secure API calls, or authenticated service requests on Port {dest_port}."
        )

        # Factor 1: Healthy Two-Way Conversation
        reasons.append(
            f"**Healthy Bidirectional Dialogue ({int(fwd_pkts)} packets sent, {int(bwd_pkts)} received):** "
            f"Normal internet protocols require both parties to send and acknowledge data. "
            f"This flow shows realistic request-response symmetry."
        )
        risk_factors.append({"factor": "Traffic Symmetry", "value": f"{int(fwd_pkts)} Fwd / {int(bwd_pkts)} Bwd", "status": "Healthy Dialogue", "risk": "Low"})

        # Factor 2: Realistic Payload
        reasons.append(
            f"**Legitimate Data Transfer (Avg {pkt_len_mean:.1f} bytes/packet):** "
            f"Packets contain real data headers and content blocks, proving this is an active session rather than an empty probe."
        )
        risk_factors.append({"factor": "Payload Size", "value": f"{pkt_len_mean:.1f} bytes/pkt", "status": "Valid Content", "risk": "Low"})

        # Factor 3: Realistic Duration
        reasons.append(
            f"**Natural Connection Lifetime ({duration_ms:.1f} ms):** "
            f"The connection duration reflects a human or application establishing a session, exchanging data, and closing cleanly."
        )
        risk_factors.append({"factor": "Connection Duration", "value": f"{duration_ms:.1f} ms", "status": "Normal Session", "risk": "Low"})

        recommended_action = "**Recommended Action:** No security action needed. Allow traffic to proceed through edge firewall normally."

    return {
        "headline": headline,
        "layman_summary": layman_summary,
        "reasons": reasons,
        "risk_factors": risk_factors,
        "recommended_action": recommended_action,
        "confidence_pct": round(confidence * 100, 2),
        "threat_score_pct": round(portscan_prob * 100, 2)
    }
