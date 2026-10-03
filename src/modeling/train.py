# src/modeling/train.py
import pandas as pd
from src.modeling.dataset import load_tokenizer_and_model, build_examples, tokenize_dataset
from src.data.theme_taxonomy import CLUSTER_TO_THEME
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

THEMES = sorted(set(CLUSTER_TO_THEME.values()))
tok, model = load_tokenizer_and_model(THEMES)

train_df = pd.read_parquet("data/processed/train.parquet")
val_df = pd.read_parquet("data/processed/val.parquet")

train_ds = tokenize_dataset(build_examples(train_df), tok)
val_ds = tokenize_dataset(build_examples(val_df), tok)

print(f"train: {len(train_ds)} examples, val: {len(val_ds)} examples")

args = TrainingArguments(
    output_dir="models/aragpt2-quotes-finetuned",
    per_device_train_batch_size=8,      # drop to 4 if you hit OOM
    per_device_eval_batch_size=8,
    num_train_epochs=8,                  # data is short (mean 24 tok), early stopping will cut this down
    learning_rate=5e-5,
    warmup_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,                  # keep best + most recent only, saves disk
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=25,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

trainer.save_model("models/aragpt2-quotes-finetuned/final")
tok.save_pretrained("models/aragpt2-quotes-finetuned/final")