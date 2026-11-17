from datetime import datetime
from urllib.parse import urlparse
import whois
from utils.dns_check import is_safe_public_host

def get_domain_age(url: str, timeout: float = 3.0) -> int:
    """Retrieve domain age in days with SSRF check and failure fallback (-1)."""
    try:
        norm = url if url.startswith(("http://", "https://")) else "http://" + url
        hostname = urlparse(norm).netloc.split(":")[0]
        if not hostname or not is_safe_public_host(hostname):
            return -1

        info = whois.whois(hostname)
        creation = info.creation_date
        if isinstance(creation, list):
            creation = creation[0]

        if creation is None:
            return -1

        age = (datetime.now() - creation).days
        return max(age, 0)
    except Exception:
        return -1