"""
Scan URL Page - Primary Landing Interface.
Clean, modern security scanner with instant SHAP attribution and structural breakdown.
"""

import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from utils.dns_check import check_domain
from utils.explainability import explain_single_url
from utils.feature_extraction import FEATURE_NAMES, extract_features
from utils.predictor import predict_url
from utils.ssl_check import check_ssl
from utils.whois_check import get_domain_age
from utils.ui import (
    apply_custom_css,
    get_threshold_config,
    get_verdict_info,
    parse_url_components,
    render_shap_reasons_chart,
    render_threat_gauge,
    render_verdict_card,
)

SAMPLE_URLS = [
    ("✅ Safe · Google", "https://www.google.com/search?q=cybersecurity"),
    ("⚠️ Spoofed brand", "http://paypal.com.account-verify.xyz/login.php?cmd=auth"),
    ("🚨 IP host · login", "http://192.168.1.1/admin/bank/signin.html"),
]


def show_scan_page():
    apply_custom_css()
    cfg = get_threshold_config()
    default_th = float(cfg.get("default_threshold", 0.52))
    high_rec_th = float(cfg.get("high_recall_threshold", 0.32))

    # Initialize session state for scan persistence
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []
    if "current_scan" not in st.session_state:
        st.session_state.current_scan = None
    if "input_url_val" not in st.session_state:
        st.session_state.input_url_val = ""

    # Header / Hero
    st.markdown(
        """
        <div class="security-header">
            <span class="security-badge">🛡️ CyberShield · Threat Analysis</span>
            <span style="font-size:0.85rem; opacity:0.75;">URL intelligence · Offline by default</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("URL Threat Inspector")
    st.caption("Analyze suspicious links for phishing signals before you open them.")

    # URL Input & Controls
    user_url = st.text_input(
        "Target URL",
        value=st.session_state.input_url_val,
        placeholder="e.g. https://secure-login.bank-portal.xyz/account",
    )

    # Clickable Example Chips
    st.caption("Try an example")
    chip_cols = st.columns(len(SAMPLE_URLS))
    for i, (label, sample_u) in enumerate(SAMPLE_URLS):
        if chip_cols[i].button(label, key=f"chip_{i}", width="stretch"):
            st.session_state.input_url_val = sample_u
            user_url = sample_u
            st.rerun()

    trigger_scan = st.button("Scan URL", type="primary", width="stretch")

    # Threshold selection (in an expander to maintain clean UI)
    with st.expander("⚙️ Detection Sensitivity & Operating Threshold", expanded=False):
        th_mode = st.radio(
            "Operating Point",
            [
                f"Default Balanced (Threshold = {default_th:.2f})",
                f"High-Recall Priority (Threshold = {high_rec_th:.2f} | OOF Recall 95.1%; Holdout 94.4%)",
                "Custom Threshold",
            ],
            index=0,
            horizontal=True,
        )
        if "Default" in th_mode:
            active_th = default_th
        elif "High-Recall" in th_mode:
            active_th = high_rec_th
        else:
            active_th = st.slider("Threshold Slider", 0.05, 0.95, default_th, 0.01)

    # Process Scan
    if trigger_scan:
        clean_url = user_url.strip()
        if not clean_url:
            st.warning("⚠️ Please provide a URL string before scanning.")
            return
        if len(clean_url) > 2048:
            st.error("❌ URL exceeds maximum permissible length (2,048 characters).")
            return
        candidate = clean_url if "://" in clean_url else "https://" + clean_url
        valid_host = False
        try:
            parsed_input = urlparse(candidate)
            valid_host = bool(parsed_input.hostname and "." in parsed_input.hostname)
            parsed_input.port
        except ValueError:
            valid_host = False
        if not valid_host or any(ch.isspace() for ch in clean_url) or parsed_input.scheme not in ("http", "https"):
            st.error("Enter a valid URL with a domain, such as https://example.com.")
            return

        try:
            with st.spinner("Analyzing URL structure…"):
                pred, prob, feats, risk = predict_url(clean_url, threshold=active_th)
                reasons = explain_single_url(clean_url, top_k=5)
                verdict = get_verdict_info(prob, low_th=high_rec_th, high_th=default_th)
        except Exception:
            st.error("We couldn’t analyze that URL. Check the address and try again.")
            return

        # Cache result to prevent re-computation on widget change
        st.session_state.current_scan = {
            "url": clean_url,
            "prediction": pred,
            "probability": prob,
            "features": feats,
            "risk_level": risk,
            "reasons": reasons,
            "verdict": verdict,
            "threshold": active_th,
        }

        # Append to session history (capped at 10)
        history_item = {
            "URL": clean_url[:48] + "..." if len(clean_url) > 48 else clean_url,
            "Verdict": f"{verdict['icon']} {verdict['label']}",
            "Score": f"{prob*100:.1f}%",
            "Risk": risk,
        }
        st.session_state.scan_history = [history_item] + [
            h for h in st.session_state.scan_history if h["URL"] != history_item["URL"]
        ][:9]

    # Display Scan Results
    res = st.session_state.current_scan
    if res is not None:
        render_verdict_card(res["verdict"], res["probability"], res["threshold"])

        render_threat_gauge(res["probability"], res["threshold"])
        st.subheader("Why this verdict")
        render_shap_reasons_chart(res["reasons"])

        # URL Breakdown
        parsed = parse_url_components(res["url"])
        with st.expander("URL breakdown", expanded=True):
            rows = [
                {"Component": "Scheme", "Value": parsed["scheme"], "Signal": "✅ HTTPS" if parsed["scheme"] == "https" else "⚠️ Not HTTPS"},
                {"Component": "Host", "Value": parsed["host"], "Signal": "🚨 IP address host" if parsed["has_ip"] else "Hostname"},
                {"Component": "Registered domain", "Value": parsed["registered_domain"], "Signal": "Registered domain"},
                {"Component": "Path", "Value": parsed["path"], "Signal": "⚠️ Redirect pattern" if parsed["has_double_slash"] else "Path text"},
                {"Component": "Query", "Value": parsed["query"], "Signal": "⚠️ Contains @ marker" if parsed["has_at"] else "Query text"},
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # Technical 40 Features
        with st.expander("Technical details · 40 lexical features", expanded=False):
            f_names = FEATURE_NAMES if len(res["features"]) == len(FEATURE_NAMES) else [f"f_{i}" for i in range(len(res["features"]))]
            feat_df = pd.DataFrame({"Feature Name": f_names, "Extracted Value": [round(v, 4) if isinstance(v, float) else v for v in res["features"]]})
            st.dataframe(feat_df, width="stretch", height=240)

        # Optional Live Network Checks
        with st.expander("Advanced · optional DNS, SSL and WHOIS checks", expanded=False):
            st.caption("Off by default. These checks contact network services and have short timeouts.")
            if st.button("Run Live Verification", key="btn_live"):
                with st.spinner("Querying DNS records..."):
                    dns_ok = check_domain(res["url"], timeout=3.0)
                with st.spinner("Checking SSL handshake..."):
                    ssl_ok = check_ssl(res["url"], timeout=3.0)
                with st.spinner("Querying WHOIS creation date..."):
                    age = get_domain_age(res["url"], timeout=3.0)

                c1, c2, c3 = st.columns(3)
                c1.metric("DNS Resolution", "Active" if dns_ok else "Unresolved / Private IP")
                c2.metric("SSL Certificate", "Valid TLS" if ssl_ok else "Self-Signed / Missing")
                c3.metric("Domain Age", f"{age} days" if age >= 0 else "Unknown / Private WHOIS")

        # Session History
        if st.session_state.scan_history:
            with st.expander("🕒 Recent Scans in this Session (Last 10)", expanded=False):
                st.dataframe(pd.DataFrame(st.session_state.scan_history), width="stretch")

    # Mandatory Legal/Technical Disclaimer
    st.info("ℹ️ This tool analyses the URL text only and never visits the site. It is not a substitute for browser or blocklist protection.")


show_scan_page()

