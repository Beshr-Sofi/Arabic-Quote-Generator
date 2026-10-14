import ast
import json
import os
import re
from collections import Counter

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.modeling.generate import generate_quotes
from src.modeling.pull_model import ensure_model

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set. Put it in .env or export it.")

# Ensure the fine-tuned model is pulled and ready locally
LOCAL_MODEL_PATH = ensure_model()

cheap = init_chat_model(os.getenv("EXTRACT_LLM", "google_genai:gemini-flash-lite-latest"))
strong = init_chat_model(os.getenv("WRITE_LLM", "google_genai:gemini-flash-latest"))

TRAIN_PATH = "data/processed/train.parquet"
MIN_FREQ = 20
MAX_TAGS = 300
DROP = set()

_JUNK = re.compile(r"^(?:widely-)?(?:mis)?attributed|^paraphrased|^quote-investigator")


def _as_list(cell) -> list[str]:
    """Parse a table cell into a list of cleaned tag strings.

    Handles literal string representations of Python lists (via `ast.literal_eval`),
    comma-separated strings, or existing iterable types, stripping whitespace from each tag.

    Args:
        cell (Any): The cell value to parse (str, list, etc.).

    Returns:
        list[str]: Cleaned list of tag strings.
    """
    if isinstance(cell, str):
        try:
            cell = ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            cell = cell.split(",")
    return [str(t).strip() for t in cell]


def load_vocab() -> list[str]:
    """Extract and rank the top tags from the training Parquet dataset.

    Aggregates tag occurrence frequencies, filters out low-frequency tags (< MIN_FREQ),
    discards attribution/junk tags matching regex `_JUNK`, and returns the top `MAX_TAGS`
    (up to 300) most common tags.

    Returns:
        list[str]: Sorted list of the top allowed vocabulary tags.

    Raises:
        AssertionError: If no valid tags are found in the training dataset.
    """
    df = pd.read_parquet(TRAIN_PATH)
    col = "themes" if "themes" in df.columns else "revised_tags"  # whichever your training used
    counts = Counter(t for cell in df[col] for t in _as_list(cell) if t)
    vocab = [t for t, c in counts.most_common()
             if c >= MIN_FREQ and not _JUNK.match(t) and t not in DROP]
    assert vocab, f"no tags found in column '{col}', check TRAIN_PATH and the column name"
    return vocab[:MAX_TAGS]


VOCAB_FILE = "data/assets/vocab.json"
VOCAB = json.load(open(VOCAB_FILE)) if os.path.exists(VOCAB_FILE) else load_vocab()
VOCAB_SET = set(VOCAB)
print(len(VOCAB), "tags:", VOCAB[:60])


# ---- step 1: Arabic statement -> 2-3 English tags the model knows
class Themes(BaseModel):
    themes: list[str] = Field(description="2-3 tags copied exactly from the allowed list")
    in_crisis: bool = Field(description="true if the statement mentions suicide, self-harm or wanting to die")

extract_chain = ChatPromptTemplate.from_messages([
    ("system", "Pick the 2 to 3 tags that best fit the user's Arabic statement. "
               "Copy them exactly from this list:\n{vocab}"),
    ("human", "{statement}"),
]) | cheap.with_structured_output(Themes)


def extract_themes(statement: str) -> tuple[list[str], bool]:
    """Extract English theme tags from an Arabic user statement and detect crisis signals.

    Uses a fast LLM with structured output:
    - If crisis/self-harm intent is detected, returns an empty tag list and True.
    - Validates predicted tags against the predefined vocabulary set (`VOCAB_SET`).
    - Retries up to 2 times, falling back to the most common vocabulary tag if extraction fails.

    Args:
        statement (str): The Arabic input text from the user.

    Returns:
        tuple[list[str], bool]: A tuple containing (list of valid theme tags, in_crisis boolean).
    """
    for _ in range(2):
        r = extract_chain.invoke({"vocab": ", ".join(VOCAB), "statement": statement})
        if r.in_crisis:
            return [], True
        valid = [t for t in dict.fromkeys(r.themes) if t in VOCAB_SET][:3]  # never trust the LLM
        if valid:
            return sorted(valid), False
    return [VOCAB[0]], False  # fallback: the most common tag


# ---- step 3: choose the best candidate AND write it as an Arabic quote
class Result(BaseModel):
    best_index: int = Field(description="0-based index of the best candidate, or -1 if none is usable")
    arabic: str = Field(description="the chosen quote as an Arabic quote, empty if best_index is -1")

write_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You edit Arabic quotes. You get the user's statement, its themes, and numbered English candidate "
     "quotes written by a small model (some are ungrammatical or meaningless).\n"
     "1. Choose the candidate that is the most coherent, complete and fitting. If none is usable, return -1.\n"
     "2. Write it in Arabic as a quote: Modern Standard Arabic, short, with rhythm and balance, natural "
     "phrasing instead of word for word. Keep the meaning. Add no new ideas, names, or religious citations."),
    ("human", "Statement: {statement}\nThemes: {themes}\nCandidates:\n{candidates}"),
]) | strong.with_structured_output(Result)


def run(statement: str) -> dict:
    """Execute the multi-stage Arabic quote generation pipeline.

    Workflow:
    1. Extracts English theme tags and checks for crisis indicators via `extract_themes`.
    2. If a crisis is indicated, returns `{'crisis': True}` for safety intervention.
    3. Generates candidate English quotes from the local fine-tuned model via `generate_quotes`.
    4. Invokes the LLM editor to select the best candidate and translate/rephrase it into
       Modern Standard Arabic with appropriate rhythm and style.
    5. Retries generation once if candidates are rejected.

    Args:
        statement (str): User's Arabic input statement or sentiment.

    Returns:
        dict: Result dictionary containing 'themes', 'english', 'arabic', 'candidates', or 'crisis'.
    """
    themes, crisis = extract_themes(statement)
    if crisis:
        return {"crisis": True}  # show a supportive message and resources, not a generated quote
    for _ in range(2):  # one retry if the editor rejects every candidate
        candidates = generate_quotes(themes, n=4, model_path=LOCAL_MODEL_PATH)
        if not candidates:
            continue
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(candidates))
        r = write_chain.invoke({"statement": statement, "themes": ", ".join(themes), "candidates": numbered})
        if 0 <= r.best_index < len(candidates) and r.arabic.strip():
            return {"themes": themes, "english": candidates[r.best_index],
                    "arabic": r.arabic.strip(), "candidates": candidates}
    return {"themes": themes, "english": None, "arabic": None, "candidates": []}


def run_safe(statement: str) -> dict:
    """Run the quote generation pipeline with error handling for API quota limits.

    Catches rate limit exceptions (HTTP 429 / RESOURCE_EXHAUSTED) from the LLM provider
    and returns a user-friendly error response.

    Args:
        statement (str): User's Arabic input statement.

    Returns:
        dict: Result dictionary from `run(statement)` or error dictionary.
    """
    try:
        return run(statement)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):  # free-tier limit reached
            return {"error": "Free daily limit reached, try again later."}
        raise