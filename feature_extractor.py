import re

def extract_signals(email_data):
    # Safely retrieve all form fields from Streamlit input
    body = email_data.get("body", "") or ""
    links = email_data.get("links", "") or ""
    sender = email_data.get("sender", "") or ""
    reply_to = email_data.get("reply_to", "") or ""
    raw_html = email_data.get("raw_html", "") or ""

    # Combine all input fields into a unified text string
    full_text = f"{sender} {reply_to} {body} {links} {raw_html}".lower()

    # Calculate engineered feature values
    length = len(full_text)
    word_count = len(full_text.split())
    url_count = len(re.findall(r'https?://\S+|www\.\S+', full_text))
    suspicious_keywords = int(any(w in full_text for w in ['urgent', 'verify', 'account', 'login', 'bank', 'password', 'update', 'security']))
    has_html = int('<html' in full_text or '<a ' in full_text or '<div' in full_text or bool(raw_html))
    special_char_count = len(re.findall(r'[!$?%]', full_text))

    # Return dictionary with all standard keys to satisfy app.py
    return {
        "text_length": length,
        "word_count": word_count,
        "url_count": url_count,
        "suspicious_keywords": suspicious_keywords,
        "has_html": has_html,
        "special_char_count": special_char_count,
        "body_length": len(body),
        "links_count": len(re.findall(r'https?://\S+|www\.\S+', links)),
        "has_raw_html": int(bool(raw_html)),
        "sender_mismatch": int(sender.lower() != reply_to.lower() if sender and reply_to else 0)
    }