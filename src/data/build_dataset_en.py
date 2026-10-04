# src/data/build_dataset_en.py
"""
Orchestrates the full English data pipeline: load -> normalize -> dedup ->
tag normalize -> cluster -> theme mapping -> stratified split.
Run as a script: python -m src.data.build_dataset_en
"""
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.download_en import load_english_quotes
from src.data.dedup import remove_exact_duplicates, remove_near_duplicates
from src.data.tag_clustering_en import (
    normalize_tags,
    get_tag_frequencies,
    cluster_tags,
    build_tag_to_theme_map,
    apply_theme_mapping,
)
from src.data.normalize_en import normalize_english
from src.data.theme_taxonomy_en import CLUSTER_TO_THEME, MANUAL_OVERRIDES

INTERIM_DIR = "data_en/interim"
PROCESSED_DIR = "data_en/processed"
N_CLUSTERS = 30
SEED = 42


def load_and_normalize() -> pd.DataFrame:
    df = load_english_quotes()
    df["tags"] = df["tags"].apply(lambda t: list(t))  # ensure plain list, not ndarray
    df["text"] = df["text"].apply(normalize_english)
    print(f"Loaded: {len(df)} rows")
    return df


def dedup_step(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_exact_duplicates(df, text_col="text")
    df = remove_near_duplicates(df, text_col="text", threshold=0.8)
    return df


def tag_and_theme_step(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tags"] = df["tags"].apply(normalize_tags)

    tag_freq = get_tag_frequencies(df)
    print(f"{len(tag_freq)} distinct normalized tags")

    clustered = cluster_tags(tag_freq, n_clusters=N_CLUSTERS)
    tag_to_theme = build_tag_to_theme_map(clustered, CLUSTER_TO_THEME)
    tagged = apply_theme_mapping(df, tag_to_theme, MANUAL_OVERRIDES)

    with open(f"{PROCESSED_DIR}/theme_taxonomy_en.json", "w", encoding="utf-8") as f:
        json.dump(
            {"cluster_to_theme": CLUSTER_TO_THEME, "overrides": MANUAL_OVERRIDES},
            f, ensure_ascii=False, indent=2,
        )
    print(tagged["themes"].explode().value_counts())
    return tagged


def fold_rare(d: pd.DataFrame, min_count: int) -> pd.DataFrame:
    counts = d["strat_theme"].value_counts()
    rare = counts[counts < min_count].index
    if len(rare) > 0:
        print(f"Folding {len(rare)} rare theme(s) into 'other': {list(rare)}")
        d = d.copy()
        d.loc[d["strat_theme"].isin(rare), "strat_theme"] = "other"
    return d


def split_step(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1):
    df = df.copy()
    df["strat_theme"] = df["themes"].apply(lambda ts: ts[0])
    df = fold_rare(df, min_count=10)

    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), random_state=SEED, stratify=df["strat_theme"]
    )
    temp_df = fold_rare(temp_df, min_count=2)

    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, random_state=SEED, stratify=temp_df["strat_theme"]
    )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(split)} rows")
        split.drop(columns=["strat_theme"]).to_parquet(f"{PROCESSED_DIR}/{name}.parquet")

    return train_df, val_df, test_df


def main():
    df = load_and_normalize()
    deduped = dedup_step(df)
    tagged = tag_and_theme_step(deduped)
    split_step(tagged)
    print("English data pipeline complete.")


if __name__ == "__main__":
    main()