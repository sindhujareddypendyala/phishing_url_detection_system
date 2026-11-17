import socket
import ipaddress
from urllib.parse import urlparse

def is_safe_public_host(hostname: str) -> bool:
    """Validate that hostname resolves to a safe, public IP address (SSRF protection)."""
    if not hostname:
        return False
    try:
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)
        if (
            ip_obj.is_private or
            ip_obj.is_loopback or
            ip_obj.is_link_local or
            ip_obj.is_reserved or
            ip_obj.is_multicast
        ):
            return False
        return True
    except Exception:
        return False

def check_domain(url: str, timeout: float = 3.0) -> bool:
    """Check if domain resolves via DNS, with timeout and SSRF protection."""
    try:
        norm = url if url.startswith(("http://", "https://")) else "http://" + url
        hostname = urlparse(norm).netloc.split(":")[0]
        if not hostname or not is_safe_public_host(hostname):
            return False
        
        orig_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            socket.gethostbyname(hostname)
            return True
        finally:
            socket.setdefaulttimeout(orig_timeout)
    except Exception:
        return False