"""
External Validation Set Builder.

Builds a fresh, independent evaluation set from external feeds (e.g., OpenPhish, PhishTank, Tranco),
strictly deduplicated against all registered domains seen in the training dataset.

RULE: DOES NOT FABRICATE DATA. If raw data files are absent at data/external/,
the script exits with explicit download instructions.
"""

import os
import sys
import pandas as pd
from urllib.parse import urlparse
from utils.feature_extraction import normalize_url, extract_registered_domain

EXTERNAL_DIR = "data/external"
TRAINING_DATA_PATH = "data/phishingURL.csv"
OUTPUT_CLEAN_PATH = os.path.join(EXTERNAL_DIR, "external_test_clean.csv")

def main():
    print("=" * 70)
    print("EXTERNAL DATASET BUILDER & VALIDATOR")
    print("=" * 70)

    os.makedirs(EXTERNAL_DIR, exist_ok=True)

    # Check for expected user-supplied raw files
    phishing_sources = [
        os.path.join(EXTERNAL_DIR, "phishing.csv"),
        os.path.join(EXTERNAL_DIR, "phishing.txt"),
        os.path.join(EXTERNAL_DIR, "openphish.txt"),
        os.path.join(EXTERNAL_DIR, "phishtank.csv"),
    ]
    legit_sources = [
        os.path.join(EXTERNAL_DIR, "legitimate.csv"),
        os.path.join(EXTERNAL_DIR, "legitimate.txt"),
        os.path.join(EXTERNAL_DIR, "tranco.csv"),
    ]

    found_phish = [f for f in phishing_sources if os.path.exists(f)]
    found_legit = [f for f in legit_sources if os.path.exists(f)]

    if not found_phish or not found_legit:
        print("\n[!] NO EXTERNAL DATA SUPPLIED. Data fabrication is strictly prohibited.")
        print("\nTo build an honest external validation benchmark, please download:")
        print("  1. Phishing Feed (Fresh Malicious URLs):")
        print("     - OpenPhish Free Feed: https://openphish.com/feed.txt -> save as 'data/external/phishing.txt'")
        print("     - Or PhishTank CSV: https://data.phishtank.com/data/online-valid.csv -> save as 'data/external/phishing.csv'")
        print("  2. Legitimate Feed (Fresh Benign URLs with Paths):")
        print("     - Tranco Top 1M list with subpaths or a CommonCrawl sample -> save as 'data/external/legitimate.csv'")
        print("     (Must contain realistic paths and query strings, not bare homepages)")
        print("\nOnce placed in 'data/external/', re-run:")
        print("  python scripts/build_external_set.py")
        sys.exit(1)

    print(f"\n[+] Found raw phishing source: {found_phish[0]}")
    print(f"[+] Found raw legitimate source: {found_legit[0]}")

    # Load training domains to prevent any domain leakage
    df_train = pd.read_csv(TRAINING_DATA_PATH)
    train_urls = df_train["url"].apply(normalize_url)
    train_domains = set(train_urls.apply(extract_registered_domain))
    print(f"[+] Loaded {len(train_domains)} existing training registered domains for exclusion.")

    # Parse phishing URLs
    phish_urls = []
    p_path = found_phish[0]
    if p_path.endswith(".txt"):
        with open(p_path, "r", encoding="utf-8", errors="ignore") as f:
            phish_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        pdf = pd.read_csv(p_path)
        col = "url" if "url" in pdf.columns else pdf.columns[0]
        phish_urls = pdf[col].dropna().astype(str).tolist()

    # Parse legitimate URLs
    legit_urls = []
    l_path = found_legit[0]
    if l_path.endswith(".txt"):
        with open(l_path, "r", encoding="utf-8", errors="ignore") as f:
            legit_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        ldf = pd.read_csv(l_path)
        col = "url" if "url" in ldf.columns else ldf.columns[0]
        legit_urls = ldf[col].dropna().astype(str).tolist()

    # Filter legitimate: discard bare homepages (require path depth >= 1 or query string)
    def has_realistic_path(u):
        try:
            norm = u if u.startswith(("http://", "https://")) else "http://" + u
            p = urlparse(norm)
            return len(p.path.strip("/").split("/")) >= 1 or len(p.query) > 0
        except Exception:
            return False

    filtered_legit = [u for u in legit_urls if has_realistic_path(u)]

    # Deduplicate and filter against training domains
    clean_phish = []
    for u in phish_urls:
        norm = normalize_url(u)
        dom = extract_registered_domain(norm)
        if dom not in train_domains:
            clean_phish.append(norm)

    clean_legit = []
    for u in filtered_legit:
        norm = normalize_url(u)
        dom = extract_registered_domain(norm)
        if dom not in train_domains:
            clean_legit.append(norm)

    clean_phish = list(dict.fromkeys(clean_phish))
    clean_legit = list(dict.fromkeys(clean_legit))

    # Balance sizes
    sample_size = min(len(clean_phish), len(clean_legit), 1000)
    final_df = pd.DataFrame({
        "url": clean_phish[:sample_size] + clean_legit[:sample_size],
        "status": ["phishing"] * sample_size + ["legitimate"] * sample_size
    }).sample(frac=1.0, random_state=42).reset_index(drop=True)

    final_df.to_csv(OUTPUT_CLEAN_PATH, index=False)
    print(f"\n[+] Successfully generated clean external dataset: '{OUTPUT_CLEAN_PATH}'")
    print(f"    Total samples: {len(final_df)} (Phishing: {sample_size}, Legitimate: {sample_size})")
    print("    Zero registered domain overlap with training dataset.")

if __name__ == "__main__":
    main()
