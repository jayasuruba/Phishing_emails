import re

def extract_signals(email_data):
    # Retrieve user input fields safely
    body = email_data.get("body", "") or ""
    links = email_data.get("links", "") or ""
    sender = email_data.get("sender", "") or ""
    
    # Combine inputs into a single text representation
    full_text = f"{sender} {body} {links}".lower()

    return [
        len(full_text),                                                     # Character count
        len(full_text.split()),                                             # Word count
        len(re.findall(r'https?://\S+|www\.\S+', full_text)),               # Link count
        int(any(w in full_text for w in ['urgent', 'verify', 'account', 'login', 'bank', 'password', 'update', 'security'])), # Keywords
        int('<html' in full_text or '<a ' in full_text or '<div' in full_text), # HTML tags
        len(re.findall(r'[!$?%]', full_text))                               # Special chars
    ]