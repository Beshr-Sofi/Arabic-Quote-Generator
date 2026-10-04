# src/data/tag_clustering_en.py
"""English-language tag clustering: normalize, frequency count, embed,
cluster, and map to topic/emotion taxonomy. Mirrors the Arabic version
in tag_clustering.py but drops all Arabic-specific normalization."""

import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter


def normalize_tag(tag: str) -> str:
    """English tag normalization: lowercase, strip punctuation/extra
    whitespace. Much lighter than the Arabic version — no script-specific
    character unification needed."""
    tag = tag.strip().lower()
    tag = re.sub(r"[^\w\s-]", "", tag)   # drop stray punctuation
    tag = re.sub(r"\s+", " ", tag).strip()
    return tag


def normalize_tags(tags: list[str]) -> list[str]:
    normalized = [normalize_tag(t) for t in tags]
    normalized = [t for t in normalized if t]       # drop empties
    return sorted(set(normalized))                   # de-dup within-row


def get_tag_frequencies(df: pd.DataFrame, tags_col: str = "tags", min_freq: int = 1) -> Counter:
    all_tags = [t for tags in df[tags_col] for t in tags if t.strip()]
    freq = Counter(all_tags)
    if min_freq > 1:
        freq = Counter({t: c for t, c in freq.items() if c >= min_freq})
    return freq


def cluster_tags(
    tag_freq: Counter,
    n_clusters: int = 30,
    model_name: str = "all-MiniLM-L6-v2",  # lighter/faster than the multilingual model — fine for English-only
) -> pd.DataFrame:
    tags = list(tag_freq.keys())
    model = SentenceTransformer(model_name)
    embeddings = model.encode(tags, show_progress_bar=True)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    result = pd.DataFrame({
        "tag": tags,
        "frequency": [tag_freq[t] for t in tags],
        "cluster_id": labels,
    }).sort_values(["cluster_id", "frequency"], ascending=[True, False])
    return result


def build_tag_to_theme_map(
    clustered: pd.DataFrame,
    cluster_to_theme: dict[int, str],
) -> dict[str, str]:
    return {
        row.tag: cluster_to_theme[row.cluster_id]
        for row in clustered.itertuples()
        if row.cluster_id in cluster_to_theme
    }


def apply_theme_mapping(
    df: pd.DataFrame,
    tag_to_theme: dict[str, str],
    overrides: dict[str, str | None],
    tags_col: str = "tags",
) -> pd.DataFrame:
    """Same resolution logic as the Arabic version: overrides win,
    None means exclude the row, unmapped-but-present tags are ignored,
    and an unresolved row falls back to 'other'."""
    def resolve_row(tags: list[str]) -> list[str] | None:
        themes: set[str] = set()
        for t in tags:
            if t in overrides:
                mapped = overrides[t]
                if mapped is None:
                    return None
                themes.add(mapped)
            elif t in tag_to_theme:
                themes.add(tag_to_theme[t])
        return sorted(themes) if themes else ["other"]

    df = df.copy()
    df["themes"] = df[tags_col].apply(resolve_row)
    excluded_count = df["themes"].isna().sum()
    print(f"Excluding {excluded_count} rows due to explicit tag exclusion")
    df = df[df["themes"].notna()].reset_index(drop=True)
    return df