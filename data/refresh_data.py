"""Download and cache every dataset used by the course, so notebooks run offline.

Run once (needs internet):  python data/refresh_data.py
Afterwards set NLP_OFFLINE=1 and every notebook reads only the files in data/.

Caches:
  * NLTK resources: punkt_tab (English sentence tokenizer), stopwords (English), wordnet
    -> data/nltk_data/   (downloaded straight from the nltk_data GitHub repo)
  * 20 Newsgroups, 4-category subset (headers/footers/quotes removed)
    -> data/20newsgroups_4cat.csv.gz
  * SMS Spam Collection (UCI), if it downloads
    -> data/sms_spam.csv.gz
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path
import urllib.request

import pandas as pd

DATA = Path(__file__).resolve().parent
NLTK_DIR = DATA / "nltk_data"
CATEGORIES = ["rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.misc"]
SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


NLTK_BASE = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"
# (zip path, keep only these sub-paths after unzipping or None = keep zip as-is)
NLTK_RESOURCES = {
    "tokenizers/punkt_tab": ["punkt_tab/english/", "punkt_tab/README"],
    "corpora/stopwords": ["stopwords/english", "stopwords/README"],
    "corpora/wordnet": None,  # NLTK reads corpora/wordnet.zip directly
}
UA = {"User-Agent": "Mozilla/5.0 (practical-nlp-course data refresh)"}


def _get(url: str) -> bytes:
    """Download with urllib; fall back to curl (some CDNs reject urllib's redirects)."""
    try:
        req = urllib.request.Request(url, headers=UA)
        return urllib.request.urlopen(req, timeout=120).read()
    except Exception as exc:
        import shutil
        import subprocess
        if not shutil.which("curl"):
            raise
        print(f"urllib failed for {url} ({exc}); retrying with curl")
        return subprocess.run(["curl", "-sSfL", "--max-time", "300", url],
                              check=True, capture_output=True).stdout


def cache_nltk() -> list[str]:
    """Fetch the NLTK packages directly (only the English parts are kept, to stay small)."""
    got = []
    for res, keep in NLTK_RESOURCES.items():
        kind = res.split("/")[0]
        dest = NLTK_DIR / kind
        dest.mkdir(parents=True, exist_ok=True)
        raw = _get(f"{NLTK_BASE}/{res}.zip")
        if keep is None:
            (dest / (res.split("/")[1] + ".zip")).write_bytes(raw)
        else:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for name in z.namelist():
                    if any(name.startswith(k) for k in keep) and not name.endswith("/"):
                        out = dest / name
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(z.read(name))
        got.append(res.split("/")[1])
    return got


NEWSGROUPS_URL = "https://ndownloader.figshare.com/files/5975967"  # 20news-bydate.tar.gz (same file sklearn uses)


def cache_newsgroups() -> dict:
    """Read the 'bydate' archive and clean it exactly like
    fetch_20newsgroups(remove=("headers", "footers", "quotes"))."""
    import tarfile
    from sklearn.datasets._twenty_newsgroups import (
        strip_newsgroup_footer, strip_newsgroup_header, strip_newsgroup_quoting)

    raw = _get(NEWSGROUPS_URL)
    rows = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = m.name.split("/")
            if not m.isfile() or len(parts) != 3 or parts[1] not in CATEGORIES:
                continue
            text = tar.extractfile(m).read().decode("latin-1")
            text = strip_newsgroup_quoting(strip_newsgroup_footer(strip_newsgroup_header(text)))
            subset = "train" if parts[0].endswith("train") else "test"
            rows.append({"doc_id": int(parts[2]), "text": text,
                         "category": parts[1], "subset": subset})
    df = (pd.DataFrame(rows).sort_values(["subset", "category", "doc_id"])
          .sample(frac=1.0, random_state=42).reset_index(drop=True))
    df.to_csv(DATA / "20newsgroups_4cat.csv.gz", index=False, compression="gzip")
    return {"n_documents": len(df),
            "by_subset": df["subset"].value_counts().to_dict(),
            "by_category": df["category"].value_counts().to_dict()}


def cache_sms() -> dict | None:
    try:
        raw = _get(SMS_URL)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            text = z.read("SMSSpamCollection").decode("utf-8", errors="replace")
    except Exception as exc:  # network or format problem: skip, the course does not depend on it
        print(f"SMS Spam Collection skipped: {exc}")
        return None
    rows = [line.split("\t", 1) for line in text.splitlines() if "\t" in line]
    df = pd.DataFrame(rows, columns=["label", "text"])
    df.to_csv(DATA / "sms_spam.csv.gz", index=False, compression="gzip")
    return {"n_messages": len(df), "by_label": df["label"].value_counts().to_dict()}


def main() -> None:
    prov = {
        "generated_on": date.today().isoformat(),
        "nltk_resources": {
            "resources": cache_nltk(),
            "source": "NLTK Data (https://www.nltk.org/nltk_data/)",
            "local_path": "data/nltk_data/",
        },
        "20newsgroups_4cat": {
            "source": "20 Newsgroups, bydate version (http://qwone.com/~jason/20Newsgroups/), "
                      "same archive sklearn.datasets.fetch_20newsgroups downloads",
            "attribution": "Collected by Ken Lang (Lang, 1995, 'NewsWeeder: Learning to Filter Netnews').",
            "categories": CATEGORIES,
            "preprocessing": "headers, footers and quoted replies removed",
            "local_path": "data/20newsgroups_4cat.csv.gz",
            **cache_newsgroups(),
        },
    }
    sms = cache_sms()
    if sms:
        prov["sms_spam"] = {
            "source": f"UCI Machine Learning Repository, SMS Spam Collection ({SMS_URL})",
            "attribution": "Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. (2011). "
                           "Contributions to the Study of SMS Spam Filtering. ACM DocEng'11.",
            "local_path": "data/sms_spam.csv.gz",
            **sms,
        }
    (DATA / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
