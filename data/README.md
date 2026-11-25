# Data

Every dataset the course uses is cached here so the notebooks run **offline**
(CI sets `NLP_OFFLINE=1`, which forbids any download). To rebuild the cache:

```bash
python data/refresh_data.py
```

| File | What it is | Size |
|---|---|---|
| `20newsgroups_4cat.csv.gz` | 20 Newsgroups, 4 categories (`rec.sport.baseball`, `sci.space`, `comp.graphics`, `talk.politics.misc`), bydate train/test split, headers/footers/quoted replies removed. Columns: `doc_id, text, category, subset` | 3,729 documents (2,239 train / 1,490 test) |
| `sms_spam.csv.gz` | SMS Spam Collection. Columns: `label, text` | 5,574 messages (4,827 ham / 747 spam) |
| `nltk_data/` | NLTK resources: `punkt_tab` (English sentence tokenizer), `stopwords` (English list), `wordnet` (for lemmatization) | ~11 MB |

`provenance.json` records the source, counts and date for each file.

## Attribution

- **20 Newsgroups** — collected by Ken Lang; see Lang, K. (1995). *NewsWeeder: Learning to
  Filter Netnews*. ICML. Distributed at http://qwone.com/~jason/20Newsgroups/ (the same
  archive `sklearn.datasets.fetch_20newsgroups` uses).
- **SMS Spam Collection** — Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011).
  *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. ACM DocEng'11.
  UCI Machine Learning Repository.
- **NLTK data** — from the NLTK Data project (https://www.nltk.org/nltk_data/).
  `wordnet.zip` is Princeton WordNet 3.0 (Miller, 1995; Fellbaum, 1998), kept byte-for-byte
  as NLTK distributes it, including Princeton's own notice inside the archive.
