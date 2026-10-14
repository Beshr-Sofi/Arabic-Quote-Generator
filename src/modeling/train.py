import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "gpt2"
MAX_LEN = 128

TRAIN_PATH = "data_en/processed/train.parquet"
EVAL_PATH = "data_en/processed/validation.parquet"

OUTPUT_DIR = "./results/gpt2-en"
FINAL_MODEL_DIR = "models/gpt2-en/final_model"

BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
SEED = 42

def build_examples(df: pd.DataFrame) -> Dataset:
    """Convert a pandas DataFrame into a Hugging Face Dataset of themes and quotes.

    Filters out missing and empty quotes, joins list of revised tags into
    a single space-separated string, and creates a Hugging Face Dataset.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'Quote' and 'revised_tags'.

    Returns:
        Dataset: Hugging Face Dataset with 'themes' and 'quote' columns.
    """
    df = df.dropna(subset=['Quote', 'revised_tags'])
    df = df[df['Quote'].str.strip() != ""]
    
    return Dataset.from_dict({
        'themes': [' '.join(row.revised_tags) for row in df.itertuples()],
        'quote': [str(row.Quote).strip() for row in df.itertuples()]
    })

def load_tokenizer_and_model(model_name: str):
    """Load pretrained causal language model and tokenizer from Hugging Face.

    Sets pad_token to eos_token since standard GPT-2 lacks a dedicated pad token.

    Args:
        model_name (str): Hugging Face model identifier (e.g. 'gpt2').

    Returns:
        tuple[AutoTokenizer, AutoModelForCausalLM]: Configured tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model

def tokenize_dataset(examples, tokenizer, max_length):
    """Tokenize and mask dataset examples for prompt-completion causal LM training.

    Formats each row into:
        Prompt: "Themes: {themes}\nQuote: "
        Target: "{quote}<|endoftext|>"
    Masks the prompt tokens in 'labels' using -100 so cross-entropy loss is computed
    only on the generated quote tokens, not the prompt.

    Args:
        examples (Dataset): Hugging Face Dataset containing 'themes' and 'quote'.
        tokenizer (AutoTokenizer): Pretrained tokenizer instance.
        max_length (int): Maximum sequence length for truncation.

    Returns:
        Dataset: Tokenized Dataset with 'input_ids', 'attention_mask', and masked 'labels'.
    """
    def preprocess_function(batch):
        inputs = [f"Themes: {t}\nQuote: " for t in batch["themes"]]
        targets = [f"{q}{tokenizer.eos_token}" for q in batch["quote"]]

        model_inputs = tokenizer(
            [i + t for i, t in zip(inputs, targets)],
            max_length=max_length,
            truncation=True
        )

        labels = []
        for i in range(len(model_inputs["input_ids"])):
            prompt_length = len(tokenizer(inputs[i])["input_ids"])
            label = model_inputs["input_ids"][i].copy()
            safe_length = min(prompt_length, len(label))
            label[:safe_length] = [-100] * safe_length
            labels.append(label)

        model_inputs["labels"] = labels
        return model_inputs

    return examples.map(
        preprocess_function,
        batched=True,
        remove_columns=["themes", "quote"]
    )

def train(
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator
) -> Trainer:
    """Configure training parameters, execute training loop, and save the final model.

    Sets up Hugging Face TrainingArguments (epochs, learning rate, early stopping,
    checkpoint saving, eval loss metric, fp16 acceleration if CUDA is available).
    After training completes, saves model weights and tokenizer to FINAL_MODEL_DIR.

    Args:
        model (AutoModelForCausalLM): Model to train.
        train_dataset (Dataset): Tokenized training dataset.
        eval_dataset (Dataset): Tokenized evaluation dataset.
        tokenizer (AutoTokenizer): Tokenizer instance.
        data_collator (DataCollatorForSeq2Seq): Dynamic padding collator for sequences and labels.

    Returns:
        Trainer: The Hugging Face Trainer instance post-training.
    """
    print("\nCreating training configuration...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting training...\n")
    trainer.train()

    print(f"\nSaving model to: {FINAL_MODEL_DIR}")
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("\nTraining completed.")
    return trainer

def all_together(
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LEN,
    train_path: str = TRAIN_PATH,
    eval_path: str = EVAL_PATH,
) -> Trainer:
    """End-to-end training pipeline orchestrator.

    Loads Parquet files, formats examples, tokenizes with label masking,
    initializes the model and tokenizer, and launches the training process.

    Args:
        model_name (str, optional): Base model name/path. Defaults to MODEL_NAME ('gpt2').
        max_length (int, optional): Max token length. Defaults to MAX_LEN (128).
        train_path (str, optional): Path to train Parquet. Defaults to TRAIN_PATH.
        eval_path (str, optional): Path to validation Parquet. Defaults to EVAL_PATH.

    Returns:
        Trainer: The completed Trainer instance.
    """

    train_examples = build_examples(pd.read_parquet(train_path))
    eval_examples = build_examples(pd.read_parquet(eval_path))

    tokenizer, model = load_tokenizer_and_model(model_name)

    print("\nTokenizing training dataset...")
    train_dataset = tokenize_dataset(train_examples, tokenizer, max_length)

    print("Tokenizing evaluation dataset...")
    eval_dataset = tokenize_dataset(eval_examples, tokenizer, max_length)

    # Use Seq2Seq collator to handle padding the -100 labels properly
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    print("\n" + "=" * 60)
    print("TOKENIZED DATASET")
    print("=" * 60)
    print(train_dataset[0])

    return train(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )