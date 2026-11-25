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
