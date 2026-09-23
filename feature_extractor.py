import re
import math
import pandas as pd
from urllib.parse import urlparse

URGENCY_WORDS = [
    "urgent", "verify", "suspend", "immediately", "action required",
    "password", "bank", "account", "confirm", "expire", "click here",
    "login", "security alert", "unusual activity", "limited time",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "buff.ly", "is.gd", "rebrand.ly", "cutt.ly",
}

FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "protonmail.com", "icloud.com", "mail.com",
}

SUSPICIOUS_TLDS = {"zip", "review", "country", "kim", "cricket", "science", "work", "party", "gq", "top", "xyz"}

HIDDEN_PATTERNS = [
    r"display\s*:\s*none",
    r"font-size\s*:\s*0(px)?",
    r"color\s*:\s*(#fff(fff)?|white)",
    r"visibility\s*:\s*hidden",
    r"opacity\s*:\s*0",
]


def _split_links(raw):
    if not raw:
        return []
    parts = re.split(r"[\s,;]+", raw.strip())
    return [p for p in parts if p]


def _domain_of(url):
    try:
        netloc = urlparse(url if "://" in url else "http://" + url).netloc
        return netloc.lower().split(":")[0]
    except Exception:
        return ""


def extract_signals(email):
    features = {}

    body = str(email.get("body", "") or "")
    links_raw = str(email.get("links", "") or "")
    sender = str(email.get("sender_domain", "") or "").lower().strip()
    reply_to = str(email.get("reply_to_domain", "") or "").lower().strip()
    raw_html = str(email.get("raw_html", "") or "")

    # 1. CONTENT SIGNALS
    body_lower = body.lower()
    features["urgency_count"] = min(sum(body_lower.count(w) for w in URGENCY_WORDS), 20)
    features["body_length_log"] = math.log1p(len(body))
    features["exclamation_count"] = min(body.count("!"), 20)
    letters = [c for c in body if c.isalpha()]
    features["caps_ratio"] = (
        sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    )
    features["body_is_empty"] = 1 if len(body.strip()) == 0 else 0

    # 2. LINK SIGNALS
    links = _split_links(links_raw)
    features["link_count"] = min(len(links), 30)
    features["has_ip_in_link"] = int(any(
        re.search(r"://\d{1,3}(\.\d{1,3}){3}", l) for l in links
    ))
    features["has_at_in_link"] = int(any("@" in _domain_of(l) or "@" in l.split("?")[0] for l in links))
    features["has_shortener"] = int(any(_domain_of(l) in SHORTENERS for l in links))
    features["suspicious_tld_count"] = sum(
        1 for l in links if _domain_of(l).rsplit(".", 1)[-1] in SUSPICIOUS_TLDS
    )
    features["avg_link_length"] = (
        sum(len(l) for l in links) / len(links) if links else 0.0
    )

    # 3. SENDER / HEADER SIGNALS
    features["sender_freemail"] = int(sender in FREEMAIL_DOMAINS)
    features["header_mismatch"] = int(
        bool(sender) and bool(reply_to) and sender != reply_to
    )
    features["sender_link_mismatch"] = int(
        bool(sender) and links and not any(
            _domain_of(l).endswith(sender) for l in links
        )
    )

    # 4. HIDDEN / OBFUSCATION SIGNALS
    features["novel_hidden_text_score"] = sum(
        len(re.findall(p, raw_html, re.I)) for p in HIDDEN_PATTERNS
    )
    features["html_tag_count"] = min(len(re.findall(r"<[^>]+>", raw_html)), 200)

    return features
