"""About CyberShield: pipeline, dataset, stack, and practical limits."""

import json
from pathlib import Path

import streamlit as st

from utils.ui import apply_custom_css


def show_about_page():
    apply_custom_css()
    st.title("About CyberShield")
    st.caption("A URL text analyzer that helps surface common phishing patterns.")

    st.subheader("How an analysis works")
    st.markdown(
        """
        **URL text** → **normalization** → **lexical and character-pattern features**
        → **trained classifier** → **probability, verdict, and SHAP explanation**

        The classifier uses URL structure and text signals. It does not open the destination page.
        """
    )

    reports = Path("reports/metrics.json")
    dataset = {}
    if reports.exists():
        try:
            dataset = json.loads(reports.read_text(encoding="utf-8")).get("dataset_info", {})
        except (OSError, json.JSONDecodeError):
            dataset = {}

    st.subheader("Dataset")
    st.markdown(
        f"The project uses **{dataset.get('raw_samples', 11_430):,}** raw labeled URLs, "
        f"deduplicated to **{dataset.get('deduplicated_samples', 11_256):,}** URLs across "
        f"**{dataset.get('unique_registered_domains', 6_634):,}** registered domains. "
        f"Evaluation uses a domain-grouped holdout of **{dataset.get('holdout_test_samples', 2_098):,}** URLs "
        "to reduce overlap between training and evaluation domains."
    )

    st.subheader("Technology")
    left, right = st.columns(2)
    left.markdown(
        "**Detection**\n\n"
        "- LightGBM classifier\n"
        "- scikit-learn feature pipeline\n"
        "- 40 lexical features and character n-grams"
    )
    right.markdown(
        "**Application**\n\n"
        "- Streamlit interface\n"
        "- SHAP local explanations\n"
        "- Plotly charts and CSV reports"
    )

    st.subheader("Privacy and optional checks")
    st.markdown(
        "Classification runs on the URL string locally. DNS, SSL, and WHOIS lookups are optional and run only "
        "when started from the Advanced section on the scan page."
    )

    st.subheader("Limitations")
    st.markdown(
        "- URL text analysis cannot identify malicious content served from a compromised but otherwise normal-looking site.\n"
        "- Shared hosting and multi-tenant domains can produce false positives.\n"
        "- Treat the result as one security signal alongside browser protection and current blocklists."
    )


show_about_page()
