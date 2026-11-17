import socket
from urllib.parse import urlparse

def check_domain(url):

    try:

        domain = urlparse(url).netloc

        socket.gethostbyname(domain)

        return True

    except:

        return False