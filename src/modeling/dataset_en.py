import re
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"
MAX_LEN = 48


def load_tokenizer_and_model(themes: list[str]):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    theme_tokens = [f"<|theme:{t}|>" for t in themes]
    tok.add_special_tokens({"additional_special_tokens": theme_tokens})
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tok))
    return tok, model


def is_likely_mojibake_or_non_english(text: str) -> bool:
    suspicious = re.findall(r'[ØÙÃÂ]', text)
    return len(suspicious) > 2


def filter_english_only(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    before = len(df)
    mojibake = df[text_col].apply(is_likely_mojibake_or_non_english)
    non_ascii_heavy = df[text_col].apply(
        lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1) > 0.3
    )
    df = df[~mojibake & ~non_ascii_heavy]
    print(f"English-only filter: {before} -> {len(df)} ({before - len(df)} removed)")
    return df


def quality_filter(df: pd.DataFrame, text_col: str = "text", min_words: int = 3, max_words: int = 60) -> pd.DataFrame:
    before = len(df)
    text = df[text_col].str.strip()
    n_words = text.str.split().str.len()
    keep = (n_words >= min_words) & (n_words <= max_words)
    df = df[keep]
    print(f"Quality filter: {before} -> {len(df)} ({before - len(df)} removed)")
    return df


def clean_split(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply both English-only and length filters, in this order —
    cheap non-English check first, then the more expensive length logic."""
    df = filter_english_only(df, text_col)
    df = quality_filter(df, text_col)
    return df.reset_index(drop=True)


def build_examples(df: pd.DataFrame, text_col: str = "text") -> list[str]:
    examples = []
    for row in df.itertuples():
        text = getattr(row, text_col).strip()
        for theme in row.themes:
            examples.append(f"<|theme:{theme}|>{text}<|endoftext|>")
    return examples


def tokenize_dataset(examples: list[str], tok) -> Dataset:
    ds = Dataset.from_dict({"text": examples})
    def _tok(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LEN)
    return ds.map(_tok, batched=True, remove_columns=["text"])

train = pd.read_parquet("data_en/processed/train.parquet")
val = pd.read_parquet("data_en/processed/val.parquet")
test = pd.read_parquet("data_en/processed/test.parquet")

train = clean_split(train)
val = clean_split(val)
test = clean_split(test)

train.to_parquet("data_en/processed/train.parquet")
val.to_parquet("data_en/processed/val.parquet")
test.to_parquet("data_en/processed/test.parquet")