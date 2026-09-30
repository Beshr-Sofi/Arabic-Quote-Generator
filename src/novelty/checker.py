from dataclasses import dataclass
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src.data.normalize import normalize_arabic


@dataclass
class NoveltyResult:
    candidate: str
    nearest: str
    cosine: float
    edit_similarity: float
    memorized: bool


class NoveltyChecker:
    def __init__(self, cos_threshold: float = 0.85, edit_threshold: float = 0.85):
        self.cos_threshold = cos_threshold
        self.edit_threshold = edit_threshold

    def fit(self, corpus: list[str]) -> "NoveltyChecker":
        # normalize again: it's idempotent and guarantees the corpus and the
        # candidates always go through the exact same function
        self.corpus = [normalize_arabic(t) for t in corpus]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(self.corpus)
        return self

    def check(self, candidate: str) -> NoveltyResult:
        norm = normalize_arabic(candidate)
        if not norm:  # empty output counts as a failure, never as "novel"
            return NoveltyResult(candidate, "", 0.0, 0.0, True)

        vec = self.vectorizer.transform([norm])
        sims = linear_kernel(vec, self.matrix).ravel()
        idx = int(sims.argmax())
        nearest = self.corpus[idx]
        cosine = float(sims[idx])
        edit_sim = fuzz.ratio(norm, nearest) / 100

        memorized = cosine >= self.cos_threshold or edit_sim >= self.edit_threshold
        return NoveltyResult(candidate, nearest, cosine, edit_sim, memorized)

def drop_middle_word(t: str) -> str:
    w = t.split()
    return " ".join(w[: len(w) // 2] + w[len(w) // 2 + 1 :]) if len(w) > 3 else t

train = pd.read_parquet("data/processed/train.parquet")
val = pd.read_parquet("data/processed/val.parquet")
chk = NoveltyChecker().fit(train["text_normalized"].tolist())

val_scores = [chk.check(t).cosine
              for t in val["text_normalized"].sample(300, random_state=0)]
print("val (should be low):", np.percentile(val_scores, [50, 90, 95, 99]).round(2))

pert_scores = [chk.check(drop_middle_word(t)).cosine
               for t in train["text_normalized"].sample(300, random_state=0)]
print("perturbed train (should be high):", np.percentile(pert_scores, [1, 5, 10, 50]).round(2))