"""Small helpers shared by the course notebooks.

Everything is loaded from the files cached in ``data/``. When the environment variable
``NLP_OFFLINE=1`` is set (as in CI), nothing is ever downloaded: a missing file raises an
error instead. Otherwise a missing file is fetched once with ``data/refresh_data.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
NLTK_DIR = DATA / "nltk_data"
OFFLINE = os.environ.get("NLP_OFFLINE", "0") == "1"

NEWSGROUPS_FILE = DATA / "20newsgroups_4cat.csv.gz"
SMS_FILE = DATA / "sms_spam.csv.gz"


def _ensure(path: Path) -> Path:
    if path.exists():
        return path
    if OFFLINE:
        raise FileNotFoundError(f"{path} is not cached and NLP_OFFLINE=1 forbids downloading. "
                                "Run `python data/refresh_data.py` once with internet access.")
    subprocess.run([sys.executable, str(DATA / "refresh_data.py")], check=True)
    return path


def setup_nltk() -> None:
    """Point NLTK at the course's cached resources (punkt_tab, stopwords, wordnet)."""
    import nltk

    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    _ensure(NLTK_DIR / "corpora" / "wordnet.zip")


def load_newsgroups(subset: str | None = None):
    """The 4-category 20 Newsgroups subset as a DataFrame (text, category, subset, doc_id)."""
    import pandas as pd

    df = pd.read_csv(_ensure(NEWSGROUPS_FILE), compression="gzip", keep_default_na=False)
    if subset is not None:
        df = df[df["subset"] == subset].reset_index(drop=True)
    return df


def load_sms():
    """The SMS Spam Collection as a DataFrame (label, text)."""
    import pandas as pd

    return pd.read_csv(_ensure(SMS_FILE), compression="gzip", keep_default_na=False)


# ---------------------------------------------------------------------------
# Text preprocessing (built step by step in notebook 02 and reused afterwards)
# ---------------------------------------------------------------------------
import re
from functools import lru_cache

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
NON_LETTER_RE = re.compile(r"[^A-Za-z\s]")
SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove URLs, e-mail addresses, numbers and every non-letter character."""
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NUMBER_RE.sub(" ", text)
    text = NON_LETTER_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _stopwords() -> frozenset:
    setup_nltk()
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=200_000)
def _lemma(token: str) -> str:
    lem = _lemmatizer()
    return lem.lemmatize(lem.lemmatize(token, pos="n"), pos="v")


@lru_cache(maxsize=1)
def _lemmatizer():
    setup_nltk()
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def preprocess(text: str, lowercase: bool = True, remove_stopwords: bool = True,
               normalize: str | None = "lemma", min_len: int = 2,
               keep: frozenset | set = frozenset()) -> list[str]:
    """Clean -> lowercase -> tokenize -> drop stopwords/short tokens -> lemma or stem.

    normalize: "lemma" (WordNet, noun then verb), "stem" (Porter) or None.
    keep: words that must survive stopword removal (e.g. {"not", "no"}).
    """
    setup_nltk()
    from nltk.tokenize import word_tokenize

    text = clean_text(text)
    if lowercase:
        text = text.lower()
    tokens = word_tokenize(text)
    if remove_stopwords:
        stops = _stopwords()
        tokens = [t for t in tokens if t.lower() not in stops or t.lower() in keep]
    tokens = [t for t in tokens if len(t) >= min_len]
    if normalize == "lemma":
        tokens = [_lemma(t) for t in tokens]
    elif normalize == "stem":
        stem = _stemmer().stem
        tokens = [stem(t) for t in tokens]
    elif normalize is not None:
        raise ValueError("normalize must be 'lemma', 'stem' or None")
    return tokens
