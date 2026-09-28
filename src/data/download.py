import ast
import json
import pandas as pd
from datasets import load_dataset

def download_heshamharoon():
    dataset = load_dataset("HeshamHaroon/arabic-quotes", split="train")
    dataset.to_json("data/raw/arabic-quotes/train.jsonl")

def download_ahmedbou():
    dataset2 = load_dataset("AhmedBou/Arabic_Quotes", split="train")
    dataset2.to_csv("data/raw/arabic-quotes-2/train.csv")

def _split_tag_string(raw: str) -> list[str]:
    """One raw tags-list element may itself be a comma-separated string
    of multiple tags, using the Arabic comma '،' or a plain ','."""
    parts = raw.replace("،", ",").split(",")
    return [p.strip() for p in parts if p.strip()]

def load_heshamharoon(path: str = "data/raw/arabic-quotes/train.jsonl") -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tags: list[str] = []
            for t in rec.get("tags", []):
                tags.extend(_split_tag_string(t))
            rows.append({
                "text": rec["quote"].strip(),
                "tags": tags,
                "author": rec.get("author") or None,
                "source": "heshamharoon",
            })
    return pd.DataFrame(rows)

def load_ahmedbou(path: str = "data/raw/arabic-quotes-2/train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    
    def parse_tags(cell):
        try:
            raw_list = ast.literal_eval(cell)  # "['a', 'b']" -> ['a', 'b']
        except (ValueError, SyntaxError):
            raw_list = [cell] if isinstance(cell, str) else []
        tags: list[str] = []
        for t in raw_list:
            tags.extend(_split_tag_string(t))
        return tags
    
    df["tags"] = df["tags"].apply(parse_tags)
    return pd.DataFrame({
        "text": df["quote"].str.strip(),
        "tags": df["tags"],
        "author": None,
        "source": "ahmedbou",
    })

def build_merged_dataset() -> pd.DataFrame:
    download_heshamharoon()
    download_ahmedbou()
    df1 = load_heshamharoon()
    df2 = load_ahmedbou()
    merged = pd.concat([df1, df2], ignore_index=True)
    return merged