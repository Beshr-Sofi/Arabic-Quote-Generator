import re
import pandas as pd

_EN_STOPWORDS = {
    "the", "and", "of", "to", "is", "that", "you", "it", "in", "not", "are",
    "with", "for", "as", "be", "this", "have", "your", "but", "was",
}

def normalize_quotes(text: pd.Series) -> pd.Series:
    """Clean and standardize a pandas Series of raw quote texts.

    Applies a chain of regex operations to:
    1. Remove leading non-alphanumeric characters.
    2. Replace quotation marks (single, double, curly) with spaces.
    3. Strip trailing special symbols and brackets.
    4. Collapse consecutive whitespace into single spaces.
    5. Convert text to lowercase and trim leading/trailing whitespace.

    Args:
        text (pd.Series): Series containing raw quote strings.

    Returns:
        pd.Series: Series of normalized, cleaned quote strings.
    """
    return (text.str.replace(r"^[^A-Za-z0-9]+", "", regex=True)             
             .str.replace(r"""["'“”‘’]+""", " ", regex=True)
             .str.replace(r"[()[\]{}/|><&;^`~]+$", "", regex=True)   
             .str.replace(r"\s+", " ", regex=True)                     
             .str.lower()                                             
             .str.strip())

def parse_tags_list(text) -> list[str]:
    """Clean and parse comma-separated tags into lists of individual tag strings.

    Performs character cleaning (removing quotation marks, non-alphanumeric prefixes,
    trailing brackets) and splits each row on commas into a Python list of tags.

    Args:
        text (pd.Series): Series containing comma-separated raw tag strings.

    Returns:
        pd.Series: Series where each row is a list of cleaned tag strings.
    """
    return (text.str.replace(r"^[^A-Za-z0-9]+", "", regex=True)             
             .str.replace(r"""["'“”‘’]+""", "", regex=True)                 
             .str.replace(r"[()\[\]{}/|><&;^`~]+$", "", regex=True)   
             .str.replace(r"\s+", " ", regex=True)                     
             .str.lower()                                           
             .str.strip()
             .str.split(','))


def normalize_tags(tag: pd.Series, lemmatizer) -> pd.Series:
    """Parse comma-separated tags and lemmatize each word to its root form.

    Calls `parse_tags_list` and then applies an NLTK WordNetLemmatizer
    to normalize inflected word variations (e.g., 'running' -> 'run').

    Args:
        tag (pd.Series): Series containing comma-separated raw tag strings.
        lemmatizer: An instantiated lemmatizer object (e.g. NLTK WordNetLemmatizer).

    Returns:
        pd.Series: Series of lists containing cleaned, lemmatized tag strings.
    """
    tag = parse_tags_list(tag)
    tag = tag.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return tag
   
def remove_quotes_containing_nonASCII(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows with missing data, empty quotes, or non-ASCII characters.

    Ensures that only clean English quotes remain by:
    1. Dropping nulls in 'Quote' and 'revised_tags'.
    2. Removing quotes that are purely whitespace.
    3. Retaining only quotes composed entirely of standard ASCII characters.

    Args:
        df (pd.DataFrame): Dataset containing 'Quote' and 'revised_tags' columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame with non-ASCII and invalid entries removed.
    """
    df = df.dropna(subset=['Quote', 'revised_tags'])
    df = df[df['Quote'].str.strip() != ""]
    df = df[df['Quote'].map(lambda x: str(x).isascii())]
    return df


def looks_english(text: str) -> bool:
    """Check whether a text string appears to be valid English using stopword heuristics.

    Extracts alphabetical word tokens:
    - If the sentence has fewer than 6 words, it passes automatically.
    - Otherwise, counts how many recognized English stopwords (_EN_STOPWORDS) appear.
    - Requires at least 1 stopword for 6-9 words, and at least 2 stopwords for 10+ words.

    Args:
        text (str): Input text string to evaluate.

    Returns:
        bool: True if the string passes the English heuristics test, False otherwise.
    """
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 6:
        return True
    hits = sum(w in _EN_STOPWORDS for w in words)
    return hits >= (1 if len(words) < 10 else 2)
 