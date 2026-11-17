import whois

from datetime import datetime
from urllib.parse import urlparse

def get_domain_age(url):

    try:

        domain = urlparse(url).netloc

        info = whois.whois(domain)

        creation = info.creation_date

        if isinstance(
            creation,
            list
        ):
            creation = creation[0]

        age = (
            datetime.now()
            - creation
        ).days

        return age

    except:

        return -1