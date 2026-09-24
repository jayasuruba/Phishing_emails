import re

def extract_signals(email_data):
    body = email_data.get("body", "") or ""
    links = email_data.get("links", "") or ""
    sender = email_data.get("sender", "") or ""
    
    full_text = f"{sender} {body} {links}".lower()

    return {
        "text_length": len(full_text),
        "word_count": len(full_text.split()),
        "url_count": len(re.findall(r'https?://\S+|www\.\S+', full_text)),
        "suspicious_keywords": int(any(w in full_text for w in ['urgent', 'verify', 'account', 'login', 'bank', 'password', 'update', 'security'])),
        "has_html": int('<html' in full_text or '<a ' in full_text or '<div' in full_text),
        "special_char_count": len(re.findall(r'[!$?%]', full_text))
    }