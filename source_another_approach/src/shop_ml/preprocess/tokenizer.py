
import re
import unicodedata
from typing import List

# Basic multilingual-aware lowercasing and tokenization.
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", re.UNICODE)

def normalize(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    return text

def tokenize(text: str) -> List[str]:
    text = normalize(text)
    return TOKEN_RE.findall(text)

def join_name_brand(name: str, brand: str) -> str:
    name = normalize(name)
    brand = normalize(brand)
    if brand and brand not in name:
        return f"{name} [SEP] {brand}"
    return name or brand
