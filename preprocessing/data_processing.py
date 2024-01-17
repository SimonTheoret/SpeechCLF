import re
from functools import partial
from typing import Any, Callable, Optional

import contextualSpellCheck
import fire
import nltk
import pandas as pd
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

"""
initial data content:
- count: (Integer) number of users who coded each tweet
- hate_speech_annotation: (Integer) number of users who judged the tweet to be hate
  speech
- offensive_language_annotation: (Integer) number of users who judged the tweet to be
  offensive,
- neither_annotation: (Integer) number of users who judged the tweet to be neither
  offensive nor non-offensive,
- class: (Class Label) class label for majority of CF users
        0: 'hate-speech',
        1: 'offensive-language'
        2: 'neither'
- tweet: (string) """


def setup_nlp_pipeline() -> Any:
    """Sets up the default nlp pipeline for spellchecking. It returns the pipeline,
    ready to be used with `nlp_pipeline_corrector`"""
    nlp = spacy.load("en_core_web_sm")
    contextualSpellCheck.add_to_pipe(nlp)
    return nlp


def nlp_pipeline_corrector(text: str, nlp=None) -> str:
    """Uses contextualSpellCheck to return the corrected string"""
    if nlp is not None:
        doc = nlp(text)
        return doc._.outcome_spellCheck
    else:
        return text


spellcheck_corrector = partial(nlp_pipeline_corrector, nlp=setup_nlp_pipeline())


def preprocess_text(
    row: pd.Series,
    lemmatizer: WordNetLemmatizer,
    stopwords: list[str],
    corrector: Optional[Callable[[str], str]] = spellcheck_corrector,
) -> str:
    """
    This function does the following preprocessing:
    - remove the part of the string before and containing the colon
    - remove punctuation
    - remove urls and replace by URLHERE
    - remove mentions and replace by MENTIONHERE
    - tokenize and remove stopwords
    - lemmatize words
    - remove empty tokens and spaces
    - join the tokens together, separated by spaces
    - if a corrector is given, the corrector is applied to every tweet after the
      previous operations are applied

    Parameters
    ----------
    row: pd.Series
        The row in a dataframe to process
    lemmatizer: WordNetLemmatizer
        The lemmatizer used to lemmatize
    stopwords: list[str]
        List of english stopwords
    corrector: Optional[Callable[[str], str]]
        Callable applied to the text
    """
    text = row.tweet

    # remove the part of the string before and containing the colon
    contains_colon = text.find(":")

    if contains_colon != -1:
        text = text[contains_colon + 1 :]

    # remove punctuation:
    text = re.sub(r"[^\w\s]", " ", text)

    # remove urls and replace by URLHERE:
    giant_url_regex = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
        r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    text = re.sub(giant_url_regex, "URLHERE", text)

    # remove mentions and replace by MENTIONHERE:
    mention_regex = r"@[\w\-]+"

    text = re.sub(mention_regex, "MENTIONHERE", text)

    # tokenize and remove stopwords:
    tokens = word_tokenize(text)

    tokens = [token.lower() for token in tokens]

    tokens = [word for word in tokens if word not in stopwords]

    # lemmatize words:
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # remove empty tokens and spaces:
    tokens = [token for token in tokens if token != "" and token != " "]

    # join the tokens together, separated by spaces:
    text = " ".join(tokens)

    # apply correction to text
    if corrector is not None:
        text = corrector(text)

    return text


def write_cleaned_data(
    src: str = "data/labeled_data.csv", dest: str = "data/cleaned.csv"
):
    """Cleans the datframe loaded at `src` and writes the cleaned  dataframe to disk
    `dest`.
    Parameters:
    ----------
    src: str
        csv file to read from. Defaults to `data/labeled_data.csv`.
    dest: str
        csv file to write to. Defaults to `data/cleaned.csv`.
    """
    df = pd.read_csv(src, index_col=[0])

    def hate(row: pd.Series):
        """Return probabilities instead of integers"""
        return row["hate_speech"] / row["count"]

    def offensive(row: pd.Series):
        return row["offensive_language"] / row["count"]

    def neither(row: pd.Series):
        return row["neither"] / row["count"]

    df["hate_speech_proba"] = df.apply(hate, axis=1)
    df["offensive_language_proba"] = df.apply(offensive, axis=1)
    df["neither_proba"] = df.apply(neither, axis=1)
    df.drop(
        columns=["hate_speech", "offensive_language", "neither", "count", "class"],
        inplace=True,
    )
    df = df[["offensive_language_proba", "hate_speech_proba", "neither_proba", "tweet"]]

    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words("english")

    preprocess = partial(
        preprocess_text, lemmatizer=lemmatizer, stopwords=stopwords_english
    )
    tqdm.pandas(desc="my bar!")

    df["tweet"] = df.progress_apply(preprocess, axis=1)
    df.to_csv(dest)
    print(f"cleaned data written to {dest}")


def load_cleaned_data(src: str = "data/cleaned.csv"):
    """Load in memory the dataframe by reading for `src`. It formats properly the
    dataframe before loading it by specifying the index column.

    Parameters
    ----------
    src: str
        csv file to read from. Defaults to `data/cleaned.csv`.
    """
    print(f"read data .csv file at {src}")
    return pd.read_csv(src, index_col=[0])


if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    fire.Fire(write_cleaned_data)
