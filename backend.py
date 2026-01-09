import math
import re
import pickle
from collections import Counter, defaultdict

import pandas as pd
from google.cloud import storage

from inverted_index_gcp import InvertedIndex

BUCKET_NAME = "ir-project-2026-databucket"
# אם הקורס נתן לך מספר מסמכים אחר — תעדכני כאן:
CORPUS_SIZE = 6348910

def _get_stopwords():
    """Try loading NLTK stopwords; if not available, return empty set."""
    try:
        import nltk
        from nltk.corpus import stopwords
        # לא מוריד אוטומטית כדי לא לתקוע import; אם חסר, תורידי במחברת: nltk.download('stopwords')
        return frozenset(stopwords.words('english'))
    except Exception:
        return frozenset()

EN_STOPWORDS = _get_stopwords()

def tokenize(text: str):
    tokens = [m.group() for m in RE_WORD.finditer(text.lower())]
    if EN_STOPWORDS:
        tokens = [t for t in tokens if t not in EN_STOPWORDS]
    return tokens


# -------- GCS helpers --------
def _bucket():
    # עובד גם בלי project id מפורש (משתמש בהרשאות של auth.authenticate_user())
    return storage.Client().bucket(BUCKET_NAME)

def _gcs_open(path: str, mode: str):
    return _bucket().blob(path).open(mode)

def _safe_load_pickle(path: str, default):
    try:
        with _gcs_open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return default

def _safe_load_pagerank_csv_gz(path: str):
    """Loads pagerank from gzipped CSV if exists; otherwise empty dict."""
    try:
        import gzip
        with _gcs_open(path, "rb") as raw:
            with gzip.GzipFile(fileobj=raw) as gz:
                d = pd.read_csv(gz, header=None, index_col=0).squeeze("columns").to_dict()
                return d
    except Exception:
        return {}


class BackendClass:
    def __init__(self):
        """
        Minimal backend that loads indices from GCS and supports:
        search, search_body, search_title, search_anchor, get_pagerank, get_pageview.

        Notes:
        - If titles / pagerank / pageviews are missing from the bucket, graceful fallbacks are used.
        """
        print("init backend class")

        self.index_name = "index"
        self.text_idx_path = "text_stemmed"
        self.title_idx_path = "title_stemmed"
        self.anchor_idx_path = "anchor_stemmed"

        # Load indices from bucket (this is the key fix)
        self.text_index = InvertedIndex.read_index(self.text_idx_path, self.index_name, BUCKET_NAME)
        self.title_index = InvertedIndex.read_index(self.title_idx_path, self.index_name, BUCKET_NAME)
        self.anchor_index = InvertedIndex.read_index(self.anchor_idx_path, self.index_name, BUCKET_NAME)

        # Optional resources (if you have them in the bucket). If not found -> fallback.
        self.doc_id_to_title = _safe_load_pickle("id_title/id_title_dict.pkl", {})  # אם יש לך קובץ כזה
        # אופציונלי: אם אין, נחזיר "doc_<id>" ככותרת.

        # pagerank / pageviews optional
        # נסי לשים אצלך בדלי אחד מהנתיבים האלו, או תשאירי - זה יחזור 0.
        pr = _safe_load_pickle("pr/pagerank.pkl", None)
        if pr is None:
            # fallback attempt: the csv.gz path from הדוגמה שלך
            pr = _safe_load_pagerank_csv_gz("pr/part-00000-65f8552b-1b0d-4846-8d4e-74cf90eec0b7-c000.csv.gz")
        self.page_rank = pr if isinstance(pr, dict) else {}

        pv = _safe_load_pickle("pv/pageview.pkl", {})
        self.page_views = pv if isinstance(pv, dict) else {}

        self.corpus_size = CORPUS_SIZE

    # ---------- internal helpers ----------
    def _title_for(self, doc_id: int) -> str:
        # אם אין מילון כותרות, נחזיר משהו יציב במקום None
        return self.doc_id_to_title.get(doc_id, f"doc_{doc_id}")

    def _idf(self, df: int) -> float:
        # avoid division by zero
        if df <= 0:
            return 0.0
        return math.log((self.corpus_size + 1) / df, 10)

    # ---------- API methods ----------
    def search_title(self, query: str):
        """
        Returns ALL results that contain a query word in the TITLE,
        ranked by number of DISTINCT query words matched (descending).
        """
        tokens = list(set(tokenize(query)))
        scores = defaultdict(int)

        for t in tokens:
            if t in self.title_index.df:
                pl = self.title_index.read_a_posting_list(self.title_idx_path, t, BUCKET_NAME)
                for doc_id, _tf in pl:
                    scores[doc_id] += 1

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [(str(doc_id), self._title_for(doc_id)) for doc_id, _ in ranked]

    def search_anchor(self, query: str):
        """
        Returns ALL results that contain a query word in ANCHOR,
        ranked by total matched query-term counts in anchor postings (descending).
        """
        tokens = tokenize(query)
        scores = defaultdict(int)

        for t in tokens:
            if t in self.anchor_index.df:
                pl = self.anchor_index.read_a_posting_list(self.anchor_idx_path, t, BUCKET_NAME)
                for doc_id, tf in pl:
                    scores[doc_id] += int(tf)  # משתמשים בתדירות שהמילה מופיעה בעוגנים

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [(str(doc_id), self._title_for(doc_id)) for doc_id, _ in ranked]

    def search_body(self, query: str):
        """
        TF-IDF + cosine similarity over BODY (based on text_index).
        Returns up to top 100 (doc_id, title).
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)

        # build query vector weights
        q_weights = {}
        for t, tf in q_tf.items():
            df = self.text_index.df.get(t, 0)
            idf = self._idf(df)
            if idf > 0:
                q_weights[t] = (1.0 + math.log(tf, 10)) * idf

        if not q_weights:
            return []

        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))

        # accumulate doc scores (dot product) + partial doc norm on query terms
        doc_dot = defaultdict(float)
        doc_norm_sq = defaultdict(float)

        for t, wq in q_weights.items():
            pl = self.text_index.read_a_posting_list(self.text_idx_path, t, BUCKET_NAME)
            idf = self._idf(self.text_index.df.get(t, 0))

            for doc_id, tf in pl:
                wd = (1.0 + math.log(tf, 10)) * idf
                doc_dot[doc_id] += wd * wq
                doc_norm_sq[doc_id] += wd * wd

        # cosine (approx using only query-term components of doc norm)
        scores = []
        for doc_id, dot in doc_dot.items():
            dn = math.sqrt(doc_norm_sq[doc_id])
            if dn == 0 or q_norm == 0:
                continue
            scores.append((doc_id, dot / (dn * q_norm)))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:100]
        return [(str(doc_id), self._title_for(doc_id)) for doc_id, _ in top]

    def search(self, query: str):
        """
        'Best' search: for minimum we just return search_body.
        (You can later improve with title/anchor/pagerank weights.)
        """
        return self.search_body(query)

    def get_pagerank(self, page_ids):
        """
        Returns PageRank values for the given list of wiki IDs.
        If pagerank dict is missing -> returns 0.0 for each.
        """
        res = []
        for pid in page_ids:
            # מגיע לפעמים כמחרוזת
            try:
                pid_int = int(pid)
            except Exception:
                pid_int = pid
            res.append(float(self.page_rank.get(pid_int, 0.0)))
        return res

    def get_pageview(self, page_ids):
        """
        Returns PageView values for the given list of wiki IDs (August 2021).
        If pageviews dict is missing -> returns 0 for each.
        """
        res = []
        for pid in page_ids:
            try:
                pid_int = int(pid)
            except Exception:
                pid_int = pid
            res.append(int(self.page_views.get(pid_int, 0)))
        return res
