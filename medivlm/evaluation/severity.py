"""TF-IDF based severity score (Section 3.4 + A.1).

S(d) = sum over t in T_s of TF-IDF(t, d), normalised across all
generated reports to [0, 1].

The term set T_s contains severity-indicating tokens. Per Section A.1
the authors use:
    * a curated seed list of clinical severity words (severe, critical,
      unstable, emergency, shock, ...) and
    * negative tokens harvested from each corpus via NLTK's
      SentimentIntensityAnalyzer (VADER).

We implement both components here so the full severity pipeline can
run offline on already-generated reports.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_SEVERITY_TERMS: Set[str] = {
    # explicit severity modifiers
    "severe", "critical", "unstable", "emergency", "shock",
    "acute", "worsening", "deteriorating", "markedly", "marked",
    "extensive", "significant", "progressive", "large", "massive",
    # clinical red-flags in chest radiographs
    "pneumothorax", "tension", "hemothorax", "pulmonary edema", "edema",
    "consolidation", "cardiomegaly", "tracheal deviation",
    "pneumonia", "infiltrate", "infiltration", "effusion",
    "pleural effusion", "cavitation", "mass", "nodule", "opacity",
    "abscess", "sepsis", "bronchiectasis",
    "collapse", "atelectasis", "emphysema", "fibrosis",
    "aspiration", "ards", "contusion", "fracture", "rib fracture",
    # negation / abnormal findings
    "abnormal", "abnormality", "abnormalities", "concerning",
    "worrisome", "suspicious", "obstruction", "enlarged", "hypoxia",
}


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def _ensure_nltk_resources() -> None:
    """Download the small NLTK resources needed by VADER + tokenizer."""
    needed = [
        ("sentiment/vader_lexicon", "vader_lexicon"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, pkg in needed:
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass  # offline; the scorer will still work with seed terms


def _vader_negative_terms(corpus: Iterable[str], threshold: float = -0.1) -> Set[str]:
    """Return unique lowercase tokens with compound sentiment <= threshold.

    We score each token individually; those VADER rates as negative
    (compound <= threshold) are added to T_s.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError:  # pragma: no cover
        return set()
    _ensure_nltk_resources()
    analyzer = SentimentIntensityAnalyzer()
    vocab: Set[str] = set()
    for doc in corpus:
        for tok in _TOKEN_RE.findall(doc.lower()):
            vocab.add(tok)
    return {t for t in vocab if analyzer.polarity_scores(t)["compound"] <= threshold}


class SeverityScorer:
    """Fit on a corpus of (e.g.) training reports, then score new reports."""

    def __init__(
        self,
        seed_terms: Optional[Iterable[str]] = None,
        use_nltk_sentiment: bool = True,
        normalize_across_corpus: bool = True,
    ) -> None:
        self.seed_terms: Set[str] = set(seed_terms) if seed_terms else set(DEFAULT_SEVERITY_TERMS)
        self.use_nltk_sentiment = use_nltk_sentiment
        self.normalize_across_corpus = normalize_across_corpus
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._severity_terms: Set[str] = set(self.seed_terms)
        self._corpus_min: Optional[float] = None
        self._corpus_max: Optional[float] = None

    # ------------------------------------------------------------------
    def fit(self, corpus: Sequence[str]) -> "SeverityScorer":
        """Construct T_s and the IDF table on a corpus of reports."""
        corpus = [c.lower() for c in corpus if c]
        if self.use_nltk_sentiment:
            negatives = _vader_negative_terms(corpus)
            self._severity_terms = set(self.seed_terms) | negatives
        else:
            self._severity_terms = set(self.seed_terms)

        # tokenizer tuned to capture hyphenated clinical terms
        def _tok(text: str) -> List[str]:
            return _TOKEN_RE.findall(text.lower())

        self._vectorizer = TfidfVectorizer(
            tokenizer=_tok,
            token_pattern=None,
            lowercase=False,
            sublinear_tf=True,
            norm=None,
        )
        self._vectorizer.fit(corpus)
        if self.normalize_across_corpus and corpus:
            raw_scores = [self._raw_score(d) for d in corpus]
            self._corpus_min = float(min(raw_scores))
            self._corpus_max = float(max(raw_scores))
            if self._corpus_max == self._corpus_min:
                self._corpus_max = self._corpus_min + 1e-6
        return self

    # ------------------------------------------------------------------
    def _raw_score(self, doc: str) -> float:
        assert self._vectorizer is not None, "Call fit() first."
        doc = (doc or "").lower()
        vec = self._vectorizer.transform([doc])  # sparse (1, |V|)
        vocabulary = self._vectorizer.vocabulary_
        total = 0.0
        for term in self._severity_terms:
            # multi-word terms (e.g., "pleural effusion") may not be in the
            # unigram vocab. better to collapse them by summing word-level TF-IDFs.
            for part in term.split():
                idx = vocabulary.get(part)
                if idx is None:
                    continue
                total += float(vec[0, idx])
        return total

    def score(self, doc: str) -> float:
        raw = self._raw_score(doc)
        if self.normalize_across_corpus and self._corpus_max is not None:
            return float((raw - self._corpus_min) / (self._corpus_max - self._corpus_min + 1e-9))
        return float(raw)

    def batch_score(self, docs: Sequence[str]) -> List[float]:
        return [self.score(d) for d in docs]
