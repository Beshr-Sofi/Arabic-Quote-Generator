import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
from src.data.normalize import normalize_arabic

def normalize_tags(tags: list[str]) -> list[str]:
    """Normalize each tag with the same rules as quote text, then drop
    empties that can result from stripping (e.g. a tag that was pure
    punctuation) and de-duplicate within the row."""
    normalized = [normalize_arabic(t) for t in tags]
    normalized = [t for t in normalized if t]  # drop empties
    return sorted(set(normalized))              # de-dup within-row

corrections = {
    "الذكا": "الذكاء",
    "ذكا": "ذكاء",
    "أخطا": "أخطاء",
    "العطا": "العطاء",
    "قرا": "قراءة",       # verify: could also be قرأ — check context
    'قرأ': 'قراءة',
    "القرا": "القراءة",
    "بكا": "بكاء",         # verify: weeping, not a name
    "البرا": "البراءة",    # verify: innocence, not براء (a name in some contexts)
    "أصدقا": "أصدقاء",
    "الأصدقا": "الأصدقاء",
    "إغوا": "إغواء",
}

def fix_truncated_tags(tags: list[str], corrections: dict) -> list[str]:
    return [corrections.get(t, t) for t in tags]

def build_tag_to_theme_map(
    clustered: pd.DataFrame,
    cluster_to_theme: dict[int, str],
) -> dict[str, str]:
    """clustered has columns: tag, frequency, cluster_id (from cluster_tags()).
    Maps every clustered tag to its cluster's final theme name."""
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
    """
    For each row's raw tags, resolve each tag to a theme:
    - overrides take precedence over the cluster-based mapping
    - an override value of None means: exclude this row entirely
    - a tag with no mapping at all (too rare to have been clustered) is ignored
    - a row with zero resolved themes falls back to 'أخرى'
    Adds a 'themes' column (list[str]) and drops excluded rows.
    """
    def resolve_row(tags: list[str]) -> list[str] | None:
        themes: set[str] = set()
        for t in tags:
            if t in overrides:
                mapped = overrides[t]
                if mapped is None:
                    return None  # explicit exclusion wins immediately
                themes.add(mapped)
            elif t in tag_to_theme:
                themes.add(tag_to_theme[t])
            # else: tag was too rare to cluster (min_freq filter) — skip silently
        return sorted(themes) if themes else ["أخرى"]

    df = df.copy()
    df["themes"] = df[tags_col].apply(resolve_row)

    excluded_count = df["themes"].isna().sum()
    print(f"Excluding {excluded_count} rows due to explicit tag exclusion")

    df = df[df["themes"].notna()].reset_index(drop=True)
    return df

def get_tag_frequencies(df: pd.DataFrame, tags_col: str = "tags") -> Counter:
    all_tags = [t for tags in df[tags_col] for t in tags if t.strip()]
    return Counter(all_tags)

def cluster_tags(
    tag_freq: Counter,
    n_clusters: int = 12,
    min_freq: int = 2,
) -> pd.DataFrame:
    # drop hapax-legomena tags (appear once) — too rare to matter, mostly noise
    tags = [t for t, c in tag_freq.items() if c >= min_freq]

    model = SentenceTransformer("intfloat/multilingual-e5-large")
    embeddings = model.encode(tags, show_progress_bar=True)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    result = pd.DataFrame({
        "tag": tags,
        "frequency": [tag_freq[t] for t in tags],
        "cluster_id": labels,
    }).sort_values(["cluster_id", "frequency"], ascending=[True, False])
    
    return result