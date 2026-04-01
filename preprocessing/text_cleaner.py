"""
text_cleaner.py
Cleans raw text from AILA 2019 case documents and statutes.
"""

import re


def clean_text(text: str) -> str:
    """Remove noise common in Indian court judgment text."""
    # Fix common OCR artifacts
    text = text.replace('\x00', ' ')
    text = text.replace('\ufffd', ' ')

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just numbers or dashes (page markers)
    text = re.sub(r'^\s*[\d]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-_=]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Normalize common legal abbreviations spacing
    text = re.sub(r'\bS\.C\b', 'SC', text)
    text = re.sub(r'\bH\.C\b', 'HC', text)
    text = re.sub(r'\bI\.P\.C\b', 'IPC', text)
    text = re.sub(r'\bC\.r\.P\.C\b', 'CrPC', text, flags=re.IGNORECASE)
    text = re.sub(r'\bC\.P\.C\b', 'CPC', text)

    return text.strip()


def parse_case_doc(filepath: str) -> dict:
    """
    Parse a case document (C{n}.txt).
    Format: first line = case name, second = court, third = date, rest = body.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()

    # Strip empty leading lines
    lines = [l for l in lines if l.strip()]

    if not lines:
        return {}

    case_name = lines[0].strip() if len(lines) > 0 else ''
    court = lines[1].strip() if len(lines) > 1 else ''
    date = lines[2].strip() if len(lines) > 2 else ''
    body = '\n'.join(lines[3:]) if len(lines) > 3 else ''

    return {
        'case_name': clean_text(case_name),
        'court': clean_text(court),
        'date': clean_text(date),
        'body': clean_text(body),
        'full_text': clean_text('\n'.join(lines)),
    }


def parse_statute_doc(filepath: str) -> dict:
    """
    Parse a statute document (S{n}.txt).
    Format: Title: ... \n Desc: ...
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    title_match = re.search(r'Title:\s*(.+?)(?=Desc:|$)', content, re.DOTALL)
    desc_match = re.search(r'Desc:\s*(.+)', content, re.DOTALL)

    title = clean_text(title_match.group(1)) if title_match else ''
    desc = clean_text(desc_match.group(1)) if desc_match else clean_text(content)

    return {
        'title': title,
        'description': desc,
        'full_text': clean_text(f"{title}. {desc}"),
    }
