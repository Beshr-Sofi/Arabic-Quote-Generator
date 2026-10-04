from datasets import load_dataset
import pandas as pd

def load_english_quotes() -> pd.DataFrame:
    ds = load_dataset("Abirate/english_quotes", split="train")
    df = ds.to_pandas()
    # columns: quote, author, tags (list)
    df = df.rename(columns={"quote": "text"})
    df["source"] = "abirate"
    return df[["text", "tags", "author", "source"]]