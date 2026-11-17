import re

from urllib.parse import urlparse

def extract_features(url):

    parsed = urlparse(url)

    hostname = parsed.netloc

    return [

        len(url),

        len(hostname),

        1 if re.match(
            r"\d+\.\d+\.\d+\.\d+",
            hostname
        ) else 0,

        hostname.count("."),

        url.count("-"),

        url.count("@"),

        url.count("?"),

        url.count("="),

        url.count("/"),

        1 if "https" in url else 0,

        sum(
            c.isdigit()
            for c in url
        ) / max(len(url),1),

        sum(
            c.isdigit()
            for c in hostname
        ) / max(len(hostname),1),

        hostname.count("."),

        1 if "-" in hostname else 0,

        0,

        1 if any(
            short in url
            for short in [
                "bit.ly",
                "tinyurl.com",
                "goo.gl"
            ]
        ) else 0,

        sum(
            word in url.lower()
            for word in [
                "login",
                "verify",
                "secure",
                "bank",
                "update",
                "confirm"
            ]
        ),

        365
    ]