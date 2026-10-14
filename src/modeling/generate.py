import os
import re
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.getcwd(), "models", "final"))


@lru_cache(maxsize=2)
def load_model(model_path: str = None):
    """Load and cache the fine-tuned GPT-2 tokenizer and model on available device.

    Uses lru_cache to ensure model weights are loaded into memory/GPU only once.

    Args:
        model_path (str, optional): Path or Hugging Face repo of the model. Defaults to MODEL_PATH.

    Returns:
        tuple[AutoTokenizer, AutoModelForCausalLM, str]: The tokenizer, model in eval mode, and device ('cuda' or 'cpu').
    """
    path = model_path or MODEL_PATH
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device), device


def build_prompt(themes: list[str]) -> str:
    """Format a list of theme tags into the exact conditioning prompt used during training.

    Alphabetically sorts themes and joins them with commas. Note there is no space after 'Quote:'.

    Args:
        themes (list[str]): List of theme tag strings.

    Returns:
        str: Prompt string formatted as 'Themes: {theme_1}, {theme_2}\\nQuote:'.
    """
    return f"Themes: {', '.join(sorted(themes))}\nQuote:"


def clean(text: str, finished: bool) -> str:
    """Post-process and validate a generated candidate quote string.

    1. Keeps only the first line and normalizes whitespace.
    2. If generation was cut off before EOS token (finished is False) or had multiple lines:
       finds the last sentence-ending punctuation mark (.!?) and trims subsequent unfinished tokens.
    3. If finished without ending punctuation, adds a trailing period.
    4. Filters quote length: discards any quote with fewer than 5 or more than 45 words.

    Args:
        text (str): Raw generated string decoded from model tokens.
        finished (bool): True if the sequence generated the EOS token naturally.

    Returns:
        str: Cleaned quote string, or empty string if discarded as invalid/incomplete.
    """
    lines = text.strip().split("\n")
    text = " ".join(lines[0].split())
    if len(lines) > 1:
        finished = False
    if not finished:  # hit the token limit: keep only complete sentences
        ends = [m.end() for m in re.finditer(r"[.!?]+[\"')]*", text)]
        text = text[: ends[-1]] if ends else ""
    elif text and text[-1] not in ".!?\"')":
        text = text.rstrip(",;:- ") + "."
    return text if 5 <= len(text.split()) <= 45 else ""


def generate_quotes(themes: list[str], n: int = 4, oversample: int = 3, model_path: str = None) -> list[str]:
    """Generate diverse English quote candidates conditioned on theme tags.

    Encodes the prompt, samples (n * oversample) sequences using temperature, top-p,
    repetition penalty, and ngram blocking to ensure variety. Post-processes each
    candidate using `clean`, removes case-insensitive duplicates, and returns up to n quotes.

    Args:
        themes (list[str]): List of theme keywords conditioning generation.
        n (int, optional): Target number of unique quotes to return. Defaults to 4.
        oversample (int, optional): Multiplier for generating extra candidates to allow
            filtering of incomplete or duplicate quotes. Defaults to 3.
        model_path (str, optional): Path or identifier of the model. Defaults to None.

    Returns:
        list[str]: Up to n cleaned, unique candidate quote strings.
    """
    tok, model, device = load_model(model_path)
    inputs = tok(build_prompt(themes), return_tensors="pt").to(device)
    eos = tok.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=48, do_sample=True, temperature=0.75, top_p=0.9,
            repetition_penalty=1.15, no_repeat_ngram_size=3,
            num_return_sequences=n * oversample,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos,
            eos_token_id=eos,
        )
    start = inputs["input_ids"].shape[1]
    quotes, seen = [], set()
    for seq in out:
        gen = seq[start:]
        q = clean(tok.decode(gen, skip_special_tokens=True), bool((gen == eos).any()))
        if q and q.lower() not in seen:
            seen.add(q.lower())
            quotes.append(q)
    return quotes[:n]