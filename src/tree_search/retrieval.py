"""Scope before indexing; derive all ranker features from permitted text/metadata."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from numbers import Real
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

FEATURE_NAMES = ["bm25", "tfidf_cosine", "svd_cosine", "title_overlap", "body_coverage",
                 "title_phrase", "authority", "log_age", "is_current", "is_archived",
                 "is_draft", "is_chat", "query_historical", "query_manager", "document_manager",
                 "log_document_length", "query_length"]
DOCUMENT_FIELDS = ["doc_id", "tenant", "acl", "title", "body", "source", "status", "updated_day", "available_day"]
QUERY_FIELDS = ["text", "tenant", "groups", "as_of_day"]


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def permitted(document, query):
    """Fail closed on malformed context or unknown/nonfinite temporal evidence."""
    def valid_day(value):
        if isinstance(value, bool) or not isinstance(value, Real):
            return False
        try:
            return math.isfinite(float(value)) and value >= 0
        except (ValueError, TypeError, OverflowError):
            return False

    days = [document.get("available_day"), document.get("updated_day"), query.get("as_of_day")]
    if not all(valid_day(day) for day in days):
        return False
    for context, field in [(document, "acl"), (query, "groups")]:
        groups = context.get(field)
        if not isinstance(groups, (list, tuple)) or not groups or not all(isinstance(group, str) and group for group in groups):
            return False
    if not all(isinstance(context.get("tenant"), str) and context["tenant"] for context in [document, query]):
        return False
    return (document["tenant"] == query["tenant"] and
            bool(set(document["acl"]) & set(query["groups"])) and
            max(document["available_day"], document["updated_day"]) <= query["as_of_day"])


def public_document(document):
    """An allowlist, not a blacklist: qrels/family/hidden annotations never enter."""
    return {key: document[key] for key in DOCUMENT_FIELDS}


class ScopedIndex:
    def __init__(self, documents):
        self.documents = sorted([public_document(d) for d in documents], key=lambda d: d["doc_id"])
        self.words = [tokens(d["title"] + " " + d["body"]) for d in self.documents]
        self.counts = [Counter(words) for words in self.words]
        self.lengths = np.array([len(words) for words in self.words], dtype=float)
        self.average_length = float(np.mean(self.lengths))
        self.document_frequency = Counter(word for words in self.words for word in set(words))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform([d["title"] + " " + d["body"] for d in self.documents])
        components = min(16, self.tfidf.shape[0] - 1, self.tfidf.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=max(1, components), random_state=17)
        self.projected = normalize(self.svd.fit_transform(self.tfidf))

    def bm25(self, text):
        result = np.zeros(len(self.documents), dtype=float)
        n = len(self.documents)
        for word in set(tokens(text)):
            df = self.document_frequency[word]
            idf = math.log(1 + (n - df + .5) / (df + .5))
            term = np.array([counter[word] for counter in self.counts], dtype=float)
            denominator = term + 1.2 * (.25 + .75 * self.lengths / self.average_length)
            result += idf * term * 2.2 / denominator
        return result

    def features(self, query):
        text = query["text"]
        qwords = set(tokens(text))
        query_vector = self.vectorizer.transform([text])
        tfidf = (self.tfidf @ query_vector.T).toarray().ravel()
        latent = normalize(self.svd.transform(query_vector))
        semantic = (self.projected @ latent.T).ravel()
        bm25 = self.bm25(text)
        historical = float(bool(qwords & {"prior", "previous", "archived", "2024"}))
        manager = float(bool(qwords & {"manager", "leadership"}))
        out = []
        for i, doc in enumerate(self.documents):
            title = set(tokens(doc["title"]))
            body = set(tokens(doc["body"]))
            overlap = len(title & qwords) / max(1, len(qwords))
            coverage = len(body & qwords) / max(1, len(qwords))
            out.append([bm25[i], tfidf[i], semantic[i], overlap, coverage,
                        float(text.lower() in doc["title"].lower()),
                        {"policy": 1., "wiki": .6, "chat": .3, "draft": .1}.get(doc["source"], 0.),
                        math.log1p(max(0, query["as_of_day"] - doc["updated_day"])),
                        float(doc["status"] == "current"), float(doc["status"] == "archived"),
                        float(doc["status"] == "draft"), float(doc["source"] == "chat"),
                        historical, manager, float(bool((title | body) & {"manager", "leadership"})),
                        math.log1p(self.lengths[i]), len(qwords)])
        return np.asarray(out, dtype=np.float32)


class SearchFramework:
    def __init__(self, documents, per_channel=20):
        self.documents = documents
        self.per_channel = per_channel
        self.cache = {}

    def scoped_index(self, query):
        allowed = [public_document(doc) for doc in self.documents if permitted(doc, query)]
        if not allowed:
            return None
        # All permitted contents and metadata participate, so updates/revocations invalidate.
        signature = hashlib.sha256(json.dumps(sorted(allowed, key=lambda d: d["doc_id"]), sort_keys=True).encode()).hexdigest()
        if signature not in self.cache:
            self.cache[signature] = ScopedIndex(allowed)
        return self.cache[signature]

    def candidates(self, query):
        index = self.scoped_index(query)
        if index is None:
            return [], np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []
        features = index.features({key: query[key] for key in QUERY_FIELDS})
        if not np.any(features[:, 0] > 0) and not np.any(features[:, 1] > 0):
            return [], np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []
        ids = [doc["doc_id"] for doc in index.documents]
        ordered = lambda scores: sorted(range(len(scores)), key=lambda i: (-float(scores[i]), ids[i]))
        lexical, semantic = ordered(features[:, 0]), ordered(features[:, 2])
        selected = sorted(set(lexical[:self.per_channel]) | set(semantic[:self.per_channel]), key=lambda i: ids[i])
        ranks_lexical = {row: rank + 1 for rank, row in enumerate(lexical)}
        ranks_semantic = {row: rank + 1 for rank, row in enumerate(semantic)}
        rrf = [1 / (60 + ranks_lexical[i]) + 1 / (60 + ranks_semantic[i]) for i in selected]
        return [ids[i] for i in selected], features[selected], rrf


def rank(ids, scores):
    if len(ids) != len(scores) or len(set(ids)) != len(ids) or not np.all(np.isfinite(scores)):
        raise ValueError("Ranking requires unique documents and finite matching scores")
    return sorted(ids, key=lambda doc_id: (-float(scores[ids.index(doc_id)]), doc_id))


def ndcg(ranking, grades, eligible, k=5):
    if len(set(ranking)) != len(ranking) or not set(ranking).issubset(set(eligible)):
        raise ValueError("Metric ranking must contain unique eligible documents only")
    gain = lambda grade: 2 ** grade - 1
    observed = sum(gain(grades.get(doc, 0)) / math.log2(i + 2) for i, doc in enumerate(ranking[:k]))
    ideal_grades = sorted([grades.get(doc, 0) for doc in eligible], reverse=True)[:k]
    ideal = sum(gain(grade) / math.log2(i + 2) for i, grade in enumerate(ideal_grades))
    return observed / ideal if ideal else 0.


def reciprocal_rank(ranking, grades, k=10):
    return next((1 / (i + 1) for i, doc in enumerate(ranking[:k]) if grades.get(doc, 0) >= 2), 0.)
