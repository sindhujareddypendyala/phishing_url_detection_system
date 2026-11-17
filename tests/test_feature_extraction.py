import numpy as np
import pytest
from utils.feature_extraction import (
    extract_lexical_features,
    extract_features,
    extract_registered_domain,
    normalize_url,
    URLFeatureExtractor,
    FEATURE_NAMES,
)

SAMPLE_URLS = [
    # 1. Standard benign HTTPS
    "https://www.google.com",
    # 2. IP address as host
    "http://192.168.1.1/admin/login",
    # 3. URL shortener
    "http://bit.ly/3xyzLogin",
    # 4. Punycode internationalized domain
    "https://xn--80ak6aa92e.com/secure",
    # 5. Brand impersonation in subdomain with suspicious TLD (.xyz)
    "http://paypal.com.account-verify.xyz/login",
    # 6. Brand impersonation in path on malicious domain
    "http://secure-update-center.com/paypal/verification.php?cmd=update",
    # 7. Non-standard port
    "http://example.org:8080/portal/signin",
    # 8. Userinfo '@' credential symbol in URL
    "http://google.com@phishing-target.net/auth",
    # 9. Path traversal / double slash redirect
    "https://legit-site.com//redirect?dest=http://evil.com//account",
    # 10. Complex benign URL with multiple subdomains and queries
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers?search=cookie&lang=en",
    # 11. Malformed / empty URL edge case
    "",
    # 12. Broken scheme edge case
    "not_a_valid_url:::???///",
]


def test_feature_vector_length_and_order():
    """Verify that all feature extractions return exactly 40 features matching FEATURE_NAMES."""
    assert len(FEATURE_NAMES) == 40
    for u in SAMPLE_URLS:
        f_dict = extract_lexical_features(u)
        assert len(f_dict) == 40
        f_list = extract_features(u)
        assert len(f_list) == 40
        # No NaNs or Infs
        assert not any(np.isnan(f_list)), f"NaN detected in features for {u}"
        assert not any(np.isinf(f_list)), f"Inf detected in features for {u}"


def test_training_serving_feature_parity():
    """Verify training-time (transformer) and serving-time (function) extract identical features."""
    extractor = URLFeatureExtractor()
    transformed = extractor.transform(SAMPLE_URLS)
    assert transformed.shape == (len(SAMPLE_URLS), 40)
    
    for i, u in enumerate(SAMPLE_URLS):
        serving_feats = np.array(extract_features(u), dtype=np.float32)
        np.testing.assert_array_equal(
            transformed[i],
            serving_feats,
            err_msg=f"Discrepancy between training and serving features for URL: {u}",
        )


def test_ip_address_detection():
    feats = extract_lexical_features("http://192.168.1.1/admin/login")
    assert feats["has_ip_address"] == 1.0
    assert feats["is_https"] == 0.0
    assert feats["nb_phish_keywords_url"] >= 1.0


def test_https_detection():
    feats = extract_lexical_features("https://www.google.com")
    assert feats["is_https"] == 1.0
    assert feats["has_ip_address"] == 0.0
    assert feats["https_token_in_hostname"] == 0.0


def test_https_token_spoofing():
    # https in hostname when domain is spoofed
    feats = extract_lexical_features("http://https-paypal-login.com/index.html")
    assert feats["is_https"] == 0.0
    assert feats["https_token_in_hostname"] == 1.0


def test_url_shortener():
    feats = extract_lexical_features("http://bit.ly/3xyzLogin")
    assert feats["is_shortened_url"] == 1.0


def test_punycode():
    feats = extract_lexical_features("https://xn--80ak6aa92e.com/secure")
    assert feats["is_punycode"] == 1.0


def test_brand_in_subdomain_and_suspicious_tld():
    feats = extract_lexical_features("http://paypal.com.account-verify.xyz/login")
    assert feats["brand_in_subdomain"] == 1.0
    assert feats["brand_not_in_domain"] == 1.0
    assert feats["has_suspicious_tld"] == 1.0
    assert feats["nb_phish_keywords_url"] >= 2.0  # account, verify, login


def test_brand_in_path():
    feats = extract_lexical_features("http://secure-update-center.com/paypal/verification.php")
    assert feats["brand_in_path"] == 1.0
    assert feats["brand_not_in_domain"] == 1.0
    assert feats["has_prefix_suffix_hyphen"] == 1.0  # secure-update-center has hyphens


def test_non_standard_port():
    feats = extract_lexical_features("http://example.org:8080/portal/signin")
    assert feats["has_non_standard_port"] == 1.0


def test_at_symbol_detection():
    feats = extract_lexical_features("http://google.com@phishing-target.net/auth")
    assert feats["has_at_symbol"] == 1.0
    assert feats["nb_at_url"] == 1.0


def test_double_slash_redirect():
    feats = extract_lexical_features("https://legit-site.com//redirect?dest=http://evil.com//account")
    assert feats["has_double_slash_redirect"] == 1.0


def test_registered_domain_extraction():
    assert extract_registered_domain("https://sub.sub2.example.co.uk/path") == "example.co.uk"
    assert extract_registered_domain("http://paypal.com.verify.xyz") == "verify.xyz"
    assert extract_registered_domain("http://192.168.0.1/test") == "192.168.0.1"


def test_robustness_on_malformed_urls():
    # Must not raise exceptions on None, empty, or garbage strings
    for bad in ["", None, "http://", "https://", "::::", "ftp://foo/bar"]:
        feats = extract_lexical_features(bad)
        assert isinstance(feats, dict)
        assert len(feats) == 40
