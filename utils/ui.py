"""
Shared UI components and styling for CyberShield AI.
Security-product design inspired by VirusTotal and Cloudflare Radar.
Supports both light and dark modes with consistent color semantics.
"""

import os
import json
import html
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from urllib.parse import urlparse
import tldextract

_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)

CUSTOM_CSS = """
<style>
/* Modern security-product typography & card styles */
.security-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.security-header { padding: .55rem .75rem; border: 1px solid rgba(56, 189, 248, .22); border-radius: 10px; background: linear-gradient(105deg, rgba(14, 165, 233, .11), rgba(14, 165, 233, .015)); }
.brand-name, .brand-name * { color: #ffffff !important; }
.brand-name { background: #075985; border: 1px solid #38bdf8; border-radius: 9px; padding: 5px 9px; letter-spacing: -.025em; box-shadow: 0 0 22px rgba(56, 189, 248, .18); }
.security-subtitle { color: var(--text-color); opacity: .78; max-width: 720px; }
.section-heading { margin-top: 1.25rem; }
.stButton button, .stDownloadButton button { min-height: 2.7rem; }
.stTextInput input { min-height: 3rem; }
.stDataFrame { width: 100%; }
@media (max-width: 900px) {
    h1 { font-size: clamp(1.6rem, 4vw, 2.4rem) !important; line-height: 1.15; }
    .security-header { flex-wrap: wrap; }
    .verdict-title { flex-wrap: wrap; }
    .verdict-title span:last-child { margin-left: 0 !important; width: 100%; }
    [data-testid="stMetric"] { padding: .65rem; }
}
.security-badge {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 3px 10px;
    border-radius: 9999px;
    background: rgba(2, 132, 199, 0.12);
    color: #0284c7;
    border: 1px solid rgba(2, 132, 199, 0.3);
}
.verdict-card {
    border-radius: 10px;
    padding: 18px 22px;
    margin: 16px 0;
    border-left: 6px solid;
    background: var(--secondary-background-color);
    box-shadow: 0 6px 24px rgba(2, 8, 23, 0.12);
}
.verdict-card.safe {
    border-left-color: #10b981;
}
.verdict-card.suspicious {
    border-left-color: #f59e0b;
}
.verdict-card.phishing {
    border-left-color: #ef4444;
}
.verdict-title {
    font-size: 1.25rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.verdict-summary {
    font-size: 0.95rem;
    color: var(--text-color);
    opacity: 0.85;
    margin: 0;
}
.url-chip {
    display: inline-block;
    padding: 4px 10px;
    margin: 2px 4px 2px 0;
    border-radius: 6px;
    font-family: monospace;
    font-size: 0.82rem;
    background: rgba(125, 125, 125, 0.12);
    border: 1px solid rgba(125, 125, 125, 0.2);
}
.url-chip.highlight {
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.35);
    color: #ef4444;
    font-weight: 600;
}
.metric-tile {
    border-radius: 8px;
    padding: 12px 16px;
    background: var(--secondary-background-color);
    border: 1px solid rgba(125, 125, 125, 0.15);
    text-align: center;
}
.disclaimer-box {
    font-size: 0.82rem;
    padding: 10px 14px;
    border-radius: 6px;
    background: rgba(125, 125, 125, 0.08);
    border: 1px dashed rgba(125, 125, 125, 0.25);
    color: var(--text-color);
    opacity: 0.8;
    margin-top: 14px;
}
</style>
"""

def apply_custom_css():
    """Inject minimal styling with light/dark adaptive CSS variables."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_cached_pipeline():
    """Load and cache the trained production model pipeline."""
    model_path = "models/live_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def get_threshold_config():
    """Load threshold configuration with fallbacks."""
    config_path = "models/threshold_config.json"
    default_cfg = {
        "default_threshold": 0.52,
        "balanced_threshold": 0.52,
        "high_recall_threshold": 0.32
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default_cfg


def get_verdict_info(probability: float, low_th: float = 0.32, high_th: float = 0.52) -> dict:
    """
    Returns consistent verdict status, color semantics, and English summary.
    Green = Safe, Amber = Suspicious, Red = Phishing.
    """
    if probability < low_th:
        return {
            "label": "Safe",
            "risk_level": "Low",
            "css_class": "safe",
            "icon": "✅",
            "color": "#10b981",
            "summary": "Lexical patterns align with standard legitimate domains. No high-risk signals detected."
        }
    elif probability < high_th:
        return {
            "label": "Suspicious",
            "risk_level": "Medium",
            "css_class": "suspicious",
            "icon": "⚠️",
            "color": "#f59e0b",
            "summary": "Elevated risk signals detected (e.g. uncommon token counts, elevated entropy, or subdomain depth)."
        }
    else:
        return {
            "label": "Phishing",
            "risk_level": "High",
            "css_class": "phishing",
            "icon": "🚨",
            "color": "#ef4444",
            "summary": "High-confidence phishing indicators detected (e.g. brand impersonation tokens, deceptive path keywords)."
        }


def render_verdict_card(verdict: dict, probability: float, threshold: float):
    """Render the central tri-state security verdict card."""
    pct = probability * 100
    st.markdown(
        f"""
        <div class="verdict-card {verdict['css_class']}">
            <div class="verdict-title" style="color: {verdict['color']};">
                <span>{verdict['icon']}</span>
                <span>{verdict['label'].upper()}</span>
            <span style="font-size: 0.85rem; font-weight: normal; opacity: 0.85; margin-left: auto;">
                    Risk: <b>{verdict['risk_level']}</b> &nbsp;|&nbsp; Phishing probability: <b>{pct:.1f}%</b>
                </span>
            </div>
            <p class="verdict-summary">{verdict['summary']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_threat_gauge(probability: float, threshold: float):
    """Render a threat indicator gauge with clean security-tool coloring."""
    pct = probability * 100
    color = "#ef4444" if probability >= threshold else ("#f59e0b" if probability >= 0.32 else "#10b981")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 28, "color": color}},
            title={"text": "Phishing Probability Score", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 32], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [32, 52], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [52, 100], "color": "rgba(239, 68, 68, 0.15)"}
                ],
                "threshold": {
                    "line": {"color": "#64748b", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100
                }
            }
        )
    )
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"}
    )
    st.plotly_chart(fig, width="stretch")


def render_shap_reasons_chart(reasons: list):
    """Render horizontal bar chart for top-5 SHAP decision drivers."""
    if not reasons:
        st.info("No localized attribution available for this input.")
        return

    df = pd.DataFrame(reasons)
    # SHAP value > 0 pushes towards phishing (red), < 0 towards safe (green)
    df["color"] = df["shap_value"].apply(lambda v: "#ef4444" if v >= 0 else "#10b981")
    df["direction"] = df["shap_value"].apply(lambda v: "Phishing Signal" if v >= 0 else "Benign Signal")
    st.caption("🔴 Increases phishing probability · 🟢 Decreases phishing probability")

    # Reverse order so top feature appears at the top of the horizontal bar chart
    df = df.iloc[::-1].reset_index(drop=True)

    fig = go.Figure(
        go.Bar(
            x=df["shap_value"],
            y=df["friendly_name"],
            orientation="h",
            marker=dict(color=df["color"], line=dict(width=0)),
            customdata=df["direction"],
            hovertemplate="%{y}<br>%{customdata}<br>Impact: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top Decision Drivers (SHAP Local Impact)",
        xaxis_title="SHAP Value (Impact on Phishing Probability)",
        yaxis_title="",
        height=260,
        margin=dict(l=10, r=10, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"}
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(125,125,125,0.15)")
    st.plotly_chart(fig, width="stretch")


def parse_url_components(url: str) -> dict:
    """Decompose URL into semantic structural components with anomaly flags."""
    raw = url.strip()
    norm = raw if raw.startswith(("http://", "https://")) else "http://" + raw
    try:
        p = urlparse(norm)
        ext = _TLD_EXTRACTOR(norm)
    except Exception:
        p = urlparse("")
        ext = None

    reg_domain = f"{ext.domain}.{ext.suffix}" if ext and ext.domain and ext.suffix else (p.netloc or "")
    subdomain = ext.subdomain if ext else ""

    return {
        "scheme": html.escape(p.scheme or "none"),
        "host": html.escape(p.hostname or "none"),
        "netloc": html.escape(p.netloc or "none"),
        "subdomain": subdomain,
        "registered_domain": html.escape(reg_domain),
        "path": html.escape(p.path or "/"),
        "query": html.escape(p.query or "none"),
        "has_ip": bool(p.netloc and p.netloc.replace(".", "").isdigit()),
        "has_at": "@" in raw,
        "has_double_slash": "//" in p.path
    }


def sanitize_csv_cell(value) -> str:
    """Prevent CSV formula injection by neutralizing leading formula triggers."""
    s = str(value) if value is not None else ""
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s
