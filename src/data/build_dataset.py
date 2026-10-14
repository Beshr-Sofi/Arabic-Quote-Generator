import re
import nltk
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.normalize import normalize_quotes, normalize_tags, remove_quotes_containing_nonASCII, looks_english


SEED = 42
INPUT_PATH = Path("data_en/raw/Quote_data.csv")
OUTPUT_DIR = Path("data_en/processed")

TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10

MIN_WORDS = 4
MAX_WORDS = 60
MAX_TAGS = 3

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = nltk.stem.WordNetLemmatizer()

def load_step() -> pd.DataFrame:
    """Load raw quote dataset from CSV file.

    Checks if INPUT_PATH exists and loads the CSV into a pandas DataFrame.

    Raises:
        FileNotFoundError: If INPUT_PATH does not exist on disk.

    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH.resolve()}")

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded: {len(df):,} rows")

    return df


def clean_step(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, filter, and deduplicate quotes in the dataset.

    Executes a multi-stage cleaning pipeline:
    1. Normalizes quote strings (removes unwanted symbols, lowercases).
    2. Collapses consecutive duplicate punctuation marks (e.g. '!!' -> '!').
    3. Drops non-ASCII and null quotes via `remove_quotes_containing_nonASCII`.
    4. Filters for quotes that look like valid English using `looks_english`.
    5. Restricts quote length to [MIN_WORDS, MAX_WORDS] range.
    6. Removes duplicate quotes to ensure distinct training samples.

    Args:
        df (pd.DataFrame): Raw DataFrame containing 'Quote' column.

    Returns:
        pd.DataFrame: Cleaned DataFrame with filtered quotes.
    """
    df = df.copy()

    df['Quote'] = normalize_quotes(df['Quote'])

    df['Quote'] = df['Quote'].apply(lambda x: re.sub(r'([.,!?~:/])\1+', r'\1', str(x)))

    df = remove_quotes_containing_nonASCII(df)

    df = df.loc[df['Quote'].apply(looks_english)]

    df = df.loc[(df['Quote'].str.split().str.len() >= MIN_WORDS) & (df['Quote'].str.split().str.len() <= MAX_WORDS)]

    df = df.drop_duplicates(subset="Quote",keep="first").reset_index(drop=True)

    return df


def normalize_tags_step(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and lemmatize quote tags, filtering out quotes with excessive tags.

    Normalizes the 'revised_tags' column using the WordNet lemmatizer and filters
    out rows where the number of tags exceeds `MAX_TAGS` (3).

    Args:
        df (pd.DataFrame): Cleaned DataFrame with 'revised_tags'.

    Returns:
        pd.DataFrame: DataFrame with normalized tags limited to at most MAX_TAGS.
    """
    df['revised_tags'] = normalize_tags(df['revised_tags'],lemmatizer)

    df = df.loc[df['revised_tags'].apply(lambda x: len(x) <= MAX_TAGS)]

    return df

def split_step(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets and persist to Parquet.

    Uses a two-phase train_test_split:
    - Train split: TRAIN_SIZE (80%).
    - Validation split: VAL_SIZE (10%).
    - Test split: TEST_SIZE (10%).
    Saves 'train.parquet', 'validation.parquet', and 'test.parquet' to OUTPUT_DIR.

    Args:
        df (pd.DataFrame): Fully preprocessed and cleaned DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).
    """

    df = df.reset_index(drop=True).copy()

    temporary_size = VAL_SIZE + TEST_SIZE

    train_df, temp_df = train_test_split(df,test_size=temporary_size,random_state=SEED,shuffle=True)

    relative_test_size = TEST_SIZE / temporary_size

    val_df, test_df = train_test_split(temp_df,test_size=relative_test_size,random_state=SEED,shuffle=True)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_parquet(OUTPUT_DIR / "train.parquet",index=False)
    val_df.to_parquet(OUTPUT_DIR / "validation.parquet",index=False)
    test_df.to_parquet(OUTPUT_DIR / "test.parquet",index=False)

    print("\nDataset split completed:")
    print(f"Total:      {len(df):,}")
    print(f"Train:      {len(train_df):,} ({len(train_df) / len(df):.1%})")
    print(f"Validation: {len(val_df):,} ({len(val_df) / len(df):.1%})")
    print(f"Test:       {len(test_df):,} ({len(test_df) / len(df):.1%})")

    return train_df, val_df, test_df


def main():
    """Execute the end-to-end data processing and splitting pipeline.

    Orchestrates loading, cleaning, tag normalization, and dataset splitting.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df).
    """
    df = load_step()
    df = clean_step(df)
    df = normalize_tags_step(df)

    train_df, val_df, test_df = split_step(df)
    print("\nPipeline completed successfully.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    main()
