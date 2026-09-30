from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
from src.data.theme_taxonomy import CLUSTER_TO_THEME

MODEL_NAME = "aubmindlab/aragpt2-base"

def load_tokenizer_and_model(themes: list[str]):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    theme_tokens = [f"<|theme:{t}|>" for t in themes]
    tok.add_special_tokens({"additional_special_tokens": theme_tokens})
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tok))
    return tok, model

MAX_LEN = 96

def build_examples(df: pd.DataFrame) -> list[str]:
    examples = []
    for row in df.itertuples():
        for theme in row.themes:      # a quote with multiple themes -> one example per theme
            examples.append(f"<|theme:{theme}|>{row.text.strip()}<|endoftext|>")
    return examples

def tokenize_dataset(examples: list[str], tok) -> Dataset:
    ds = Dataset.from_dict({"text": examples})
    def _tok(batch):
        out = tok(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out
    return ds.map(_tok, batched=True, remove_columns=["text"])


THEMES = sorted(set(CLUSTER_TO_THEME.values()))
train = pd.read_parquet("data/processed/train.parquet")

tok, model = load_tokenizer_and_model(THEMES)
train_examples = build_examples(train)
small = tokenize_dataset(train_examples[:200], tok)

args = TrainingArguments(
    output_dir="/tmp/dryrun", per_device_train_batch_size=4,
    num_train_epochs=1, logging_steps=5, report_to="none"
)
trainer = Trainer(model=model, args=args, train_dataset=small)
trainer.train()