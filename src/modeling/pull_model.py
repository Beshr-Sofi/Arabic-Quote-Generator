"""Utility script to pull the fine-tuned quote generation model from Hugging Face Hub

and save it into the local 'models/final' directory.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = os.getenv("HF_MODEL_REPO", "Beshr-Sofi/quote-gpt2")
DEFAULT_LOCAL_DIR = Path(os.getenv("LOCAL_MODEL_DIR", "models/final"))


def pull_model(repo_id: str = DEFAULT_REPO_ID, local_dir: Path | str = DEFAULT_LOCAL_DIR) -> str:
    """Pull the model and tokenizer from Hugging Face Hub and save them locally.

    Args:
        repo_id (str, optional): Hugging Face repository identifier. Defaults to 'Beshr-Sofi/quote-gpt2'.
        local_dir (Path | str, optional): Local directory where model files will be saved. Defaults to 'models/final'.

    Returns:
        str: Absolute path to the local model directory.
    """
    target_dir = Path(local_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pulling model '{repo_id}' from Hugging Face Hub...")
    print(f"Target local directory: {target_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            ignore_patterns=["*.git*", "README.md"],
        )
    except Exception as e:
        print(f"snapshot_download encountered an issue ({e}), falling back to transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        tok.save_pretrained(str(target_dir))
        model.save_pretrained(str(target_dir))

    print(f"Model successfully pulled and saved at: {target_dir}")
    return str(target_dir)


def is_model_saved(local_dir: Path | str = DEFAULT_LOCAL_DIR) -> bool:
    """Check whether the model weights and configuration already exist locally.

    Args:
        local_dir (Path | str, optional): Path to check. Defaults to 'models/final'.

    Returns:
        bool: True if config.json and weights (safetensors or bin) exist, False otherwise.
    """
    target_dir = Path(local_dir).resolve()
    config_file = target_dir / "config.json"
    weights_files = [
        target_dir / "model.safetensors",
        target_dir / "pytorch_model.bin",
    ]
    return config_file.exists() and any(w.exists() for w in weights_files)


def ensure_model(repo_id: str = DEFAULT_REPO_ID, local_dir: Path | str = DEFAULT_LOCAL_DIR) -> str:
    """Check if the model is downloaded locally; pull and save it if missing.

    Args:
        repo_id (str, optional): Hugging Face repository identifier.
        local_dir (Path | str, optional): Path to local directory.

    Returns:
        str: Path to the verified local model directory.
    """
    target_dir = Path(local_dir).resolve()
    if is_model_saved(target_dir):
        return str(target_dir)

    return pull_model(repo_id=repo_id, local_dir=target_dir)


if __name__ == "__main__":
    pull_model()
