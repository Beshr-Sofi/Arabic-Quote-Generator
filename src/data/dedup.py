import pandas as pd
from datasketch import MinHash, MinHashLSH

def _shingles(text: str, k: int = 3) -> set[str]:
    """Character k-shingles. Character-level (not word-level) because
    Arabic words are short and near-dups often differ by one character,
    which word-shingles would miss entirely."""
    text = text.replace(" ", "")
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def _make_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(text):
        m.update(shingle.encode("utf8"))
    return m

def remove_exact_duplicates(df: pd.DataFrame, text_col: str = "text_normalized") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    print(f"Exact dedup: {before} -> {len(df)} ({before - len(df)} removed)")
    return df

def remove_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "text_normalized",
    threshold: float = 0.8,
    num_perm: int = 128,
) -> pd.DataFrame:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    to_drop = set()

    for idx, text in df[text_col].items():
        if len(text) < 3:  # too short to shingle meaningfully
            continue
        mh = _make_minhash(text, num_perm)
        # query BEFORE inserting: finds already-seen near-duplicates of this row
        matches = lsh.query(mh)
        if matches:
            to_drop.add(idx)  # keep the first occurrence, drop this later one
        else:
            lsh.insert(str(idx), mh)
        minhashes[idx] = mh

    before = len(df)
    df = df.drop(index=to_drop).reset_index(drop=True)
    print(f"Near-dup dedup: {before} -> {len(df)} ({len(to_drop)} removed)")
    return df