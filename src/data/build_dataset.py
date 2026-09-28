# src/data/build_dataset.py
"""
Orchestrates the full day-1 pipeline: load -> normalize -> dedup ->
tag normalize -> cluster -> theme mapping -> stratified split.
Run as a script: python -m src.data.build_dataset
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.download import build_merged_dataset
from src.data.normalize import normalize_arabic
from src.data.dedup import remove_exact_duplicates, remove_near_duplicates
from src.data.tag_clustering import (
    corrections,
    fix_truncated_tags,
    normalize_tags,
    get_tag_frequencies,
    cluster_tags,
    build_tag_to_theme_map,
    apply_theme_mapping,
)
from src.data.theme_taxonomy import CLUSTER_TO_THEME, MANUAL_OVERRIDES

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
N_CLUSTERS = 35
SEED = 42


def load_and_merge() -> pd.DataFrame:
    return build_merged_dataset()


def normalize_step(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_normalized"] = df["text"].apply(normalize_arabic)
    return df


def dedup_step(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_exact_duplicates(df, text_col="text_normalized")
    df = remove_near_duplicates(df, text_col="text_normalized", threshold=0.8)
    return df


def tag_and_theme_step(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tags"] = df["tags"].apply(normalize_tags)
    df["tags"] = df["tags"].apply(lambda ts: fix_truncated_tags(ts, corrections))

    tag_freq = get_tag_frequencies(df)
    print(f"{len(tag_freq)} distinct normalized tags")

    clustered = cluster_tags(tag_freq, n_clusters=N_CLUSTERS, min_freq=2)
    tag_to_theme = build_tag_to_theme_map(clustered, CLUSTER_TO_THEME)
    tagged = apply_theme_mapping(df, tag_to_theme, MANUAL_OVERRIDES)

    with open(f"{PROCESSED_DIR}/theme_taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(
            {"cluster_to_theme": CLUSTER_TO_THEME, "overrides": MANUAL_OVERRIDES},
            f, ensure_ascii=False, indent=2,
        )
    tagged.to_parquet(f"{INTERIM_DIR}/tagged.parquet")
    print(tagged["themes"].explode().value_counts())
    return tagged


def split_step(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1):
    df = df.copy()
    df["strat_theme"] = df["themes"].apply(lambda ts: ts[0])

    def fold_rare(d: pd.DataFrame, min_count: int) -> pd.DataFrame:
        counts = d["strat_theme"].value_counts()
        rare = counts[counts < min_count].index
        if len(rare) > 0:
            print(f"Folding {len(rare)} rare theme(s) into 'أخرى': {list(rare)}")
            d = d.copy()
            d.loc[d["strat_theme"].isin(rare), "strat_theme"] = "أخرى"
        return d

    # stage 1: needs >=2 members per class overall; use a comfortable margin
    df = fold_rare(df, min_count=10)

    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), random_state=SEED, stratify=df["strat_theme"]
    )

    # stage 2: recheck on temp_df itself — a class fine at stage 1 can still
    # collapse to 1 member here since temp_df is only ~20% of the data
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
    merged = load_and_merge()
    normalized = normalize_step(merged)
    deduped = dedup_step(normalized)
    tagged = tag_and_theme_step(deduped)
    split_step(tagged)
    print("Day 1 pipeline complete.")


if __name__ == "__main__":
    main()