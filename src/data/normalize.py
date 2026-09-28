import re
import pandas as pd

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u06D6-\u06ED]")
_TATWEEL = re.compile(r"\u0640")
_ALEF_VARIANTS = re.compile(r"[إأآا]")
_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_URL = re.compile(r"http\S+|www\.\S+")
_NON_ARABIC_PUNCT = re.compile(r"[^\u0600-\u06FF\s0-9.,،؛؟!\-]")

def normalize_arabic(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = text.translate(_EASTERN_DIGITS)
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _NON_ARABIC_PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text