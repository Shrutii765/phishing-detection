"""
feature_extraction.py
---------------------
Turns a URL (plain text) into numeric features for the ML model.

IMPORTANT: The same functions are used for training AND for live predictions,
so the model always sees features calculated in exactly the same way.
This file never opens, visits, or requests any URL. It only studies the text.
"""

import ipaddress
import re
from urllib.parse import urlparse

import pandas as pd

MAX_URL_LENGTH = 2048

# Common URL shortening services (hostnames without "www.")
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "bit.do", "t.ly",
    "rb.gy", "lnkd.in", "shorte.st", "adf.ly",
}

# Words often seen in phishing URLs (a hint, never proof on its own)
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "bank",
    "confirm", "password", "paypal", "webscr", "wallet", "billing", "suspend",
]

# Final list of features, in a fixed order. Do not change the order
# after training a model, or the model will receive scrambled inputs.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_subdomains",
    "num_hyphens",
    "num_hyphens_in_host",
    "num_underscores",
    "num_at_symbols",
    "num_question_marks",
    "num_equals",
    "num_ampersands",
    "num_percent",
    "num_digits",
    "digit_ratio",
    "path_depth",
    "has_ip_host",
    "is_https",
    "has_port",
    "is_shortener",
    "suspicious_word_count",
    "has_punycode",
]


def _parse_url(url):
    """Check the URL text and split it into parts. Raises ValueError if invalid."""
    url = str(url).strip()

    if not url:
        raise ValueError("URL is empty.")
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"URL is too long (maximum {MAX_URL_LENGTH} characters).")
    if re.search(r"\s", url):
        raise ValueError("URL must not contain spaces.")
        # Remove trailing slashes: "example.com/" and "example.com" are the same page.
    # In this dataset a trailing slash is mostly seen on phishing URLs, which
    # would teach the model a shortcut that does not hold in the real world.
    url = re.sub(r"(?<=[^/:])/+$", "", url)
    # If the user typed "example.com" without http(s)://, add a temporary
    # scheme only so the parser can find the hostname.
    to_parse = url if "://" in url else "http://" + url

    try:
        parsed = urlparse(to_parse)
        port = parsed.port  # raises ValueError if the port is invalid
        host = parsed.hostname
    except ValueError as exc:
        raise ValueError("URL could not be parsed. Please check its format.") from exc

    if not host:
        raise ValueError("URL has no valid hostname.")

    return url, parsed, host, port


def _is_ip_address(host):
    """True if the hostname is an IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def validate_url(url):
    """Used by the app. Returns (True, "") or (False, "reason")."""
    try:
        _parse_url(url)
        return True, ""
    except ValueError as exc:
        return False, str(exc)


def extract_features(url):
    """
    Convert ONE url string into a dictionary of features.
    Raises ValueError if the URL is empty or malformed.
    """
    url, parsed, host, port = _parse_url(url)

    path = parsed.path or ""
    query = parsed.query or ""
    url_lower = url.lower()

    is_ip = _is_ip_address(host)
    num_digits = sum(ch.isdigit() for ch in url)

    # Simple subdomain estimate: number of dot-separated parts minus 2.
    # (It does not handle special endings like .co.uk perfectly.)
    num_subdomains = 0 if is_ip else max(len(host.split(".")) - 2, 0)

    features = {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_subdomains": num_subdomains,
        "num_hyphens": url.count("-"),
        "num_hyphens_in_host": host.count("-"),
        "num_underscores": url.count("_"),
        "num_at_symbols": url.count("@"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_ampersands": url.count("&"),
        "num_percent": url.count("%"),
        "num_digits": num_digits,
        "digit_ratio": round(num_digits / len(url), 4),
        "path_depth": len([p for p in path.split("/") if p]),
        "has_ip_host": int(is_ip),
        "is_https": int(url_lower.startswith("https://")),
        "has_port": int(port is not None),
        "is_shortener": int(host.removeprefix("www.") in SHORTENERS),
        "suspicious_word_count": sum(url_lower.count(w) for w in SUSPICIOUS_WORDS),
        "has_punycode": int("xn--" in host),
    }

    # Return in the fixed order defined by FEATURE_NAMES
    return {name: features[name] for name in FEATURE_NAMES}


def extract_features_batch(urls):
    """
    Convert MANY urls (list or pandas Series) into a DataFrame.
    Rows for invalid URLs are filled with NaN so preprocessing can drop them.
    """
    rows = []
    for u in urls:
        try:
            rows.append(extract_features(u))
        except ValueError:
            rows.append({name: float("nan") for name in FEATURE_NAMES})
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=getattr(urls, "index", None))


if __name__ == "__main__":
    # Quick self-test. Run with:  python -m src.feature_extraction
    samples = [
        "https://www.google.com",
        "http://192.168.1.1/login/verify-account.php",
        "http://paypal-secure-login.example-verify.com/update?id=123&session=abc",
        "https://bit.ly/3abcXYZ",
        "",
        "not a url with spaces",
    ]
    for s in samples:
        print("\nURL:", repr(s))
        try:
            for key, value in extract_features(s).items():
                print(f"  {key:24} {value}")
        except ValueError as err:
            print("  ERROR (expected for bad input):", err)