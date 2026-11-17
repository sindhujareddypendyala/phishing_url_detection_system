import ssl
import socket
from urllib.parse import urlparse
from utils.dns_check import is_safe_public_host

def check_ssl(url: str, timeout: float = 3.0) -> bool:
    """Validate SSL certificate with SSRF blocking and strict socket timeout."""
    try:
        norm = url if url.startswith(("http://", "https://")) else "https://" + url
        hostname = urlparse(norm).netloc.split(":")[0]
        if not hostname or not is_safe_public_host(hostname):
            return False

        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname):
                return True
    except Exception:
        return False