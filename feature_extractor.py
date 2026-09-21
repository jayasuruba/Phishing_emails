import re
import pandas as pd

def extract_signals(email):
    features = {}

    # 1. WORDING / CONTENT SIGNALS
    text = str(email.get("body", ""))
    urgency_words = ["urgent", "verify", "suspend", "immediately", "action required", "password", "bank"]
    features["urgency_count"] = sum(text.lower().count(word) for word in urgency_words)
    features["body_length"] = len(text)
    features["exclamation_count"] = text.count("!")

    # 2. LINK / URL SIGNALS
    links = str(email.get("links", ""))
    features["link_count"] = len(links.split(",")) if links else 0
    features["has_ip_in_link"] = 1 if re.search(r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', links) else 0

    # 3. SENDER / HEADER SIGNALS
    sender = str(email.get("sender_domain", "")).lower()
    reply_to = str(email.get("reply_to_domain", "")).lower()
    features["header_mismatch"] = 1 if (sender != reply_to and reply_to != "") else 0

    # 4. NOVEL FEATURE: INVISIBLE / HIDDEN TEXT SCORE
    raw_html = str(email.get("raw_html", ""))
    hidden_patterns = [
        r'display\s*:\s*none',
        r'font-size\s*:\s*0px',
        r'color\s*:\s*(#fff|#ffffff|white)',
        r'visibility\s*:\s*hidden'
    ]
    features["novel_hidden_text_score"] = sum(len(re.findall(p, raw_html, re.I)) for p in hidden_patterns)

    return features