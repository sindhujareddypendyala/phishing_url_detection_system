"""
Lexical Feature Extraction Module for Phishing URL Detection.

This module is the SINGLE SOURCE OF TRUTH for extracting lexical and statistical
features from raw URLs for both model training and live web application serving.

All features are strictly Tier 1 (lexical, structural, syntactic, and information-theoretic).
Zero external network calls, DNS queries, or socket connections are made in this module.
Robust against malformed, partial, non-string, and empty URLs.
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs
import tldextract
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Pre-instantiate offline tldextract instance (suffix_list_urls=None prevents network downloads)
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)

# List of high-risk / commonly abused Top Level Domains in phishing campaigns
SUSPICIOUS_TLDS = frozenset({
    "xyz", "top", "work", "loan", "club", "click", "country", "stream",
    "gdn", "mom", "fit", "tk", "ml", "ga", "cf", "gq", "buzz", "surfer",
    "monster", "rest", "kim", "party", "date", "racing", "win", "bid"
})

# Known URL shortener domains
URL_SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "rb.gy", "shorturl.at", "tiny.cc",
    "cutt.ly", "bl.ink", "tr.im", "v.gd"
})

# Suspicious keywords commonly found in phishing URL paths and parameters
PHISHING_KEYWORDS = (
    "login", "verify", "secure", "update", "account", "banking",
    "signin", "confirm", "password", "security", "wallet",
    "authenticate", "alert", "billing", "recover", "support",
    "validation", "service", "credential", "auth"
)

# Targeted brand names frequently impersonated in phishing attacks
TARGET_BRANDS = (
    "paypal", "apple", "google", "microsoft", "netflix", "amazon",
    "facebook", "instagram", "whatsapp", "chase", "wellsfargo",
    "binance", "coinbase", "dropbox", "ebay", "bankofamerica"
)

# Ordered feature names for numeric vector representations (40 features)
FEATURE_NAMES = [
    # 1-4: Lengths
    "length_url",
    "length_hostname",
    "length_path",
    "length_query",
    # 5-19: Character & Token Counts
    "nb_dots_url",
    "nb_dots_hostname",
    "nb_hyphens_url",
    "nb_hyphens_hostname",
    "nb_underline_url",
    "nb_slash_url",
    "nb_qm_url",
    "nb_and_url",
    "nb_eq_url",
    "nb_at_url",
    "nb_digits_url",
    "nb_digits_hostname",
    "nb_special_chars_url",
    "ratio_digits_url",
    "ratio_digits_hostname",
    # 20-22: Structural Counts
    "nb_subdomains",
    "path_depth",
    "nb_query_params",
    # 23-30: Protocol, Syntax & Security Flags
    "is_https",
    "https_token_in_hostname",
    "has_ip_address",
    "has_at_symbol",
    "has_double_slash_redirect",
    "is_punycode",
    "is_shortened_url",
    "has_non_standard_port",
    # 31-34: Information & Domain Properties
    "entropy_url",
    "entropy_hostname",
    "has_suspicious_tld",
    "has_prefix_suffix_hyphen",
    # 35-40: Phishing Keywords & Brand Impersonation Signals
    "nb_phish_keywords_url",
    "nb_phish_keywords_host",
    "nb_phish_keywords_path",
    "brand_in_subdomain",
    "brand_in_path",
    "brand_not_in_domain",
]


def normalize_url(url: str) -> str:
    """Normalize URL by trimming whitespace, lowercasing, and stripping trailing slash."""
    if not isinstance(url, str):
        url = str(url) if url is not None else ""
    return url.strip().lower().rstrip("/")


def extract_registered_domain(url: str) -> str:
    """
    Extract registered domain (domain + public suffix, e.g. 'google.com', 'example.co.uk')
    using offline tldextract. Returns IP address or hostname fallback for raw IPs or invalid URLs.
    """
    u = normalize_url(url)
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    try:
        ext = _TLD_EXTRACTOR(u)
        reg = ext.top_domain_under_public_suffix or ext.domain or urlparse(u).netloc.split(":")[0]
        return reg if reg else "unknown"
    except Exception:
        return "unknown"


def compute_shannon_entropy(string: str) -> float:
    """Compute Shannon entropy of a given character string."""
    if not string:
        return 0.0
    length = len(string)
    counts = Counter(string)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())


def get_url_components(url: str):
    """Safely parse URL into netloc, path, query, and tldextract components."""
    norm = normalize_url(url)
    to_parse = norm if norm.startswith(("http://", "https://")) else "http://" + norm
    try:
        parsed = urlparse(to_parse)
    except Exception:
        parsed = urlparse("")
    
    hostname = parsed.netloc.split(":")[0] if parsed.netloc else ""
    path = parsed.path or ""
    query = parsed.query or ""
    try:
        port = parsed.port
    except (ValueError, Exception):
        port = None
    
    try:
        tld_res = _TLD_EXTRACTOR(to_parse)
    except Exception:
        tld_res = None
        
    return norm, parsed, hostname, path, query, port, tld_res


def extract_lengths(url: str, hostname: str, path: str, query: str) -> dict:
    """Length features for URL, host, path, and query."""
    return {
        "length_url": float(len(url)),
        "length_hostname": float(len(hostname)),
        "length_path": float(len(path)),
        "length_query": float(len(query)),
    }


def extract_char_counts(url: str, hostname: str) -> dict:
    """Character frequency and ratio metrics."""
    len_url = max(len(url), 1)
    len_host = max(len(hostname), 1)
    
    digits_url = sum(c.isdigit() for c in url)
    digits_host = sum(c.isdigit() for c in hostname)
    
    # Special chars: characters that are not alphanumeric and not standard URL delimiters
    special_chars = sum(not (c.isalnum() or c in "/:?#[]@!$&'()*+,;=._-~%") for c in url)
    
    return {
        "nb_dots_url": float(url.count(".")),
        "nb_dots_hostname": float(hostname.count(".")),
        "nb_hyphens_url": float(url.count("-")),
        "nb_hyphens_hostname": float(hostname.count("-")),
        "nb_underline_url": float(url.count("_")),
        "nb_slash_url": float(url.count("/")),
        "nb_qm_url": float(url.count("?")),
        "nb_and_url": float(url.count("&")),
        "nb_eq_url": float(url.count("=")),
        "nb_at_url": float(url.count("@")),
        "nb_digits_url": float(digits_url),
        "nb_digits_hostname": float(digits_host),
        "nb_special_chars_url": float(special_chars),
        "ratio_digits_url": float(digits_url / len_url),
        "ratio_digits_hostname": float(digits_host / len_host),
    }


def extract_structural_counts(path: str, query: str, tld_res) -> dict:
    """Count subdomains, path depth, and query parameters."""
    subdomains = tld_res.subdomain if tld_res and tld_res.subdomain else ""
    nb_sub = len(subdomains.split(".")) if subdomains else 0
    
    path_depth = len([seg for seg in path.strip("/").split("/") if seg])
    
    try:
        nb_query_params = len(parse_qs(query))
    except Exception:
        nb_query_params = query.count("=") if query else 0
        
    return {
        "nb_subdomains": float(nb_sub),
        "path_depth": float(path_depth),
        "nb_query_params": float(nb_query_params),
    }


def extract_protocol_and_flags(url: str, hostname: str, path: str, port, tld_res) -> dict:
    """Binary security and syntactic indicators."""
    is_https = 1.0 if url.startswith("https://") else 0.0
    
    # https token appearing inside hostname/domain indicates domain spoofing (e.g. https-paypal.com)
    https_token_in_host = 1.0 if "https" in hostname else 0.0
    
    # IP address check (IPv4 or IPv6)
    ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    has_ip = 1.0 if re.match(ipv4_pattern, hostname) or ":" in hostname else 0.0
    
    has_at = 1.0 if "@" in url else 0.0
    
    # Check for // redirect after protocol
    url_without_proto = re.sub(r"^https?://", "", url)
    has_double_slash = 1.0 if "//" in url_without_proto else 0.0
    
    is_punycode = 1.0 if "xn--" in hostname else 0.0
    
    reg_domain = (tld_res.top_domain_under_public_suffix or tld_res.domain or "").lower() if tld_res else ""
    is_shortener = 1.0 if reg_domain in URL_SHORTENERS or hostname in URL_SHORTENERS else 0.0
    
    has_non_std_port = 1.0 if port is not None and port not in (80, 443) else 0.0
    
    return {
        "is_https": is_https,
        "https_token_in_hostname": https_token_in_host,
        "has_ip_address": has_ip,
        "has_at_symbol": has_at,
        "has_double_slash_redirect": has_double_slash,
        "is_punycode": is_punycode,
        "is_shortened_url": is_shortener,
        "has_non_standard_port": has_non_std_port,
    }


def extract_domain_properties(url: str, hostname: str, tld_res) -> dict:
    """Shannon entropy and TLD properties."""
    entropy_url = compute_shannon_entropy(url)
    entropy_host = compute_shannon_entropy(hostname)
    
    suffix = (tld_res.suffix or "").lower() if tld_res else ""
    has_suspicious_tld = 1.0 if suffix in SUSPICIOUS_TLDS else 0.0
    
    domain_part = (tld_res.domain or "").lower() if tld_res else ""
    has_hyphen_in_domain = 1.0 if "-" in domain_part else 0.0
    
    return {
        "entropy_url": float(entropy_url),
        "entropy_hostname": float(entropy_host),
        "has_suspicious_tld": has_suspicious_tld,
        "has_prefix_suffix_hyphen": has_hyphen_in_domain,
    }


def extract_keywords_and_brands(url: str, hostname: str, path: str, tld_res) -> dict:
    """Detect phishing credential keywords and brand spoofing outside the legitimate domain."""
    nb_keywords_url = sum(kw in url for kw in PHISHING_KEYWORDS)
    nb_keywords_host = sum(kw in hostname for kw in PHISHING_KEYWORDS)
    nb_keywords_path = sum(kw in path for kw in PHISHING_KEYWORDS)
    
    subdomain = (tld_res.subdomain or "").lower() if tld_res else ""
    reg_domain = (tld_res.top_domain_under_public_suffix or tld_res.domain or "").lower() if tld_res else ""
    domain_body = (tld_res.domain or "").lower() if tld_res else ""
    
    brand_in_sub = 0.0
    brand_in_pth = 0.0
    brand_outside_domain = 0.0
    
    for brand in TARGET_BRANDS:
        if brand in subdomain:
            brand_in_sub = 1.0
        if brand in path:
            # Brand in path while registered domain is NOT the brand (e.g. attacker.com/paypal/login)
            if brand != domain_body:
                brand_in_pth = 1.0
        if brand in url:
            if brand != domain_body:
                brand_outside_domain = 1.0
                
    return {
        "nb_phish_keywords_url": float(nb_keywords_url),
        "nb_phish_keywords_host": float(nb_keywords_host),
        "nb_phish_keywords_path": float(nb_keywords_path),
        "brand_in_subdomain": brand_in_sub,
        "brand_in_path": brand_in_pth,
        "brand_not_in_domain": brand_outside_domain,
    }


def extract_lexical_features(url: str) -> dict:
    """
    Extract all 40 lexical and structural features as a dictionary.
    Safe against any malformed or non-string input.
    """
    norm, parsed, hostname, path, query, port, tld_res = get_url_components(url)
    
    features = {}
    features.update(extract_lengths(norm, hostname, path, query))
    features.update(extract_char_counts(norm, hostname))
    features.update(extract_structural_counts(path, query, tld_res))
    features.update(extract_protocol_and_flags(norm, hostname, path, port, tld_res))
    features.update(extract_domain_properties(norm, hostname, tld_res))
    features.update(extract_keywords_and_brands(norm, hostname, path, tld_res))
    
    return features


def extract_features(url: str) -> list:
    """
    Extract ordered list of 40 float features.
    Maintains compatibility with inference models expecting vector inputs.
    """
    f_dict = extract_lexical_features(url)
    return [float(f_dict[name]) for name in FEATURE_NAMES]


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that transforms raw URLs into
    a 2D numpy array of lexical features (n_samples, 40).
    """
    
    def __init__(self):
        self.feature_names_ = list(FEATURE_NAMES)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if hasattr(X, "tolist"):
            urls = X.tolist()
        elif hasattr(X, "values"):
            urls = X.values.tolist()
        else:
            urls = list(X)
            
        rows = [extract_features(u) for u in urls]
        return np.array(rows, dtype=np.float32)
        
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)