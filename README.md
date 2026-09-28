# Arabic Quote Generator

A conditional text generation project that produces original Arabic
aphorisms/quotes given a theme (e.g. صبر، حكمة، نجاح) and a style
register (modern aphorism vs. classical/poetic). Built end-to-end as
an applied project following the DeepLearning.AI NLP Specialization —
covering text preprocessing, embeddings, sequence modeling, and
transformer fine-tuning on a real, low-resource language.

## What this is

The model is fine-tuned on a merged, deduplicated corpus of Arabic
quotes and poetry, conditioned on theme via control tokens, and
served through a small API and web demo. Generated quotes are
**unattributed by design** — the model is not used to produce
attributed quotes ("— أحمد شوقي") since a generative model will
confidently invent false attributions to real authors. A separate
novelty filter checks every generated output against the training
corpus and rejects near-duplicates, so the demo doesn't surface
memorized training data as if it were new.

## Why this project

Arabic is comparatively under-resourced in NLP tooling: rich
morphology, heavy dialect variation, and inconsistent diacritization
make even "simple" generation tasks non-trivial. This project treats
the Arabic-specific challenges (normalization, near-duplicate
detection, tokenizer behavior on Arabic script) as first-class
problems rather than an afterthought to an English-first pipeline.

## Tech stack

Python, PyTorch, 🤗 Transformers & Datasets, CAMeL Tools, scikit-learn,
datasketch (MinHash), FastAPI, Gradio, DVC.