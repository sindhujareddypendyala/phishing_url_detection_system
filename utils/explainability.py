"""
Explainability utility using SHAP for global and per-URL local explanations.
"""

import joblib
import numpy as np
import shap
from utils.feature_extraction import FEATURE_NAMES, extract_lexical_features

_EXPLAINER = None
_FEATURE_NAMES_ALL = None
_PIPELINE = None

def get_explainer():
    """Lazily load and cache SHAP TreeExplainer and feature names."""
    global _EXPLAINER, _FEATURE_NAMES_ALL, _PIPELINE
    if _EXPLAINER is None:
        try:
            _PIPELINE = joblib.load("models/live_model.joblib")
            feat_union = _PIPELINE.named_steps["features"]
            tfidf_vocab = feat_union.transformer_list[1][1].get_feature_names_out()
            _FEATURE_NAMES_ALL = FEATURE_NAMES + [f"char_ngram_{ng}" for ng in tfidf_vocab]
            
            # The calibrator holds the base LightGBM estimator(s)
            calibrator = _PIPELINE.named_steps["calibrator"]
            if hasattr(calibrator, "calibrated_classifiers_"):
                base_clf = calibrator.calibrated_classifiers_[0].estimator
            elif hasattr(calibrator, "estimator"):
                base_clf = calibrator.estimator
            else:
                base_clf = calibrator
                
            _EXPLAINER = shap.TreeExplainer(base_clf)
        except Exception:
            _EXPLAINER = None
    return _EXPLAINER, _FEATURE_NAMES_ALL, _PIPELINE

FEATURE_FRIENDLY_NAMES = {
    "length_url": "Total URL Length",
    "length_hostname": "Hostname Length",
    "length_path": "Path Length",
    "length_query": "Query String Length",
    "nb_dots_url": "Dot Count in URL",
    "nb_dots_hostname": "Dot Count in Hostname",
    "nb_hyphens_url": "Hyphen Count in URL",
    "nb_hyphens_hostname": "Hyphen Count in Hostname",
    "nb_slash_url": "Slash Count in URL",
    "nb_digits_url": "Digit Count in URL",
    "nb_digits_hostname": "Digit Count in Hostname",
    "ratio_digits_url": "Digit Ratio in URL",
    "ratio_digits_hostname": "Digit Ratio in Hostname",
    "nb_subdomains": "Subdomain Depth",
    "path_depth": "Path Depth",
    "nb_query_params": "Number of Query Parameters",
    "is_https": "HTTPS Protocol Scheme",
    "https_token_in_hostname": "HTTPS Spoofing in Hostname",
    "has_ip_address": "IP Address as Hostname",
    "has_at_symbol": "Credential '@' Symbol in URL",
    "has_double_slash_redirect": "Double Slash Redirect '//'",
    "is_punycode": "Punycode (xn--) Internationalized Domain",
    "is_shortened_url": "Known URL Shortener Domain",
    "has_non_standard_port": "Non-Standard Network Port",
    "entropy_url": "Shannon Entropy of URL",
    "entropy_hostname": "Shannon Entropy of Hostname",
    "has_suspicious_tld": "High-Risk / Suspicious TLD",
    "has_prefix_suffix_hyphen": "Hyphen in Domain Name",
    "nb_phish_keywords_url": "Phishing Keywords in URL",
    "nb_phish_keywords_host": "Phishing Keywords in Hostname",
    "nb_phish_keywords_path": "Phishing Keywords in Path",
    "brand_in_subdomain": "Target Brand Spoofed in Subdomain",
    "brand_in_path": "Target Brand Spoofed in Path",
    "brand_not_in_domain": "Brand Impersonation Outside Official Domain"
}

def explain_single_url(url: str, top_k: int = 5):
    """
    Compute local SHAP explanation for a single URL.
    Returns list of top_k dictionaries:
    [{"feature": ..., "friendly_name": ..., "value": ..., "impact": "Increases Risk" / "Decreases Risk", "shap_value": ...}]
    """
    explainer, all_names, pipeline = get_explainer()
    if explainer is None or pipeline is None:
        return []

    try:
        feat_union = pipeline.named_steps["features"]
        x_trans = feat_union.transform([url])
        dense_x = x_trans.toarray()
        shap_vals = explainer.shap_values(dense_x)
        
        # Binary LightGBM TreeExplainer output
        if isinstance(shap_vals, list):
            s_row = shap_vals[1][0]
        elif shap_vals.ndim == 3:
            s_row = shap_vals[0, :, 1]
        elif shap_vals.ndim == 2:
            s_row = shap_vals[0]
        else:
            s_row = shap_vals
            
        x_vals = dense_x[0]
        
        # Sort by absolute impact
        ranked_indices = np.argsort(np.abs(s_row))[::-1][:top_k]
        
        reasons = []
        for idx in ranked_indices:
            raw_name = all_names[idx] if idx < len(all_names) else f"feature_{idx}"
            friendly = FEATURE_FRIENDLY_NAMES.get(raw_name, raw_name.replace("char_ngram_", "Character Pattern '") + ("'" if "char_ngram_" in raw_name else ""))
            shap_v = float(s_row[idx])
            val = float(x_vals[idx])
            impact = "Increases Risk (Phishing Signal)" if shap_v > 0 else "Decreases Risk (Benign Signal)"
            reasons.append({
                "feature": raw_name,
                "friendly_name": friendly,
                "value": round(val, 4),
                "shap_value": round(shap_v, 4),
                "impact": impact
            })
        return reasons
    except Exception as e:
        return [{"feature": "error", "friendly_name": f"Explainability note: {e}", "value": 0, "shap_value": 0, "impact": "N/A"}]
