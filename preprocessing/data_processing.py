import re
from functools import partial

import fire
import nltk
import pandas as pd
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


def preprocess_text(
    row: pd.Series, lemmatizer: WordNetLemmatizer, stopwords: list[str]
) -> str:
    """
    Preprocessing text
    """
    text = row.tweet

    text = re.sub("<[^<]+?>", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)

    tokens = word_tokenize(text)

    tokens = [token.lower() for token in tokens]

    tokens = [word for word in tokens if word not in stopwords]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    tokens = [token for token in tokens if token != "" and token != " "]

    text = " ".join(tokens)

    return text


def write_cleaned_data(
    src: str = "data/labeled_data.csv", dest: str = "data/cleaned.csv"
):
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

    df["tweet"] = df.apply(preprocess, axis=1)
    print(df.columns)
    df.to_csv(dest)
    print(f"cleaned data written to {dest}")


def load_cleaned_data(src: str = "data/cleaned.csv"):
    print(f"read data .csv file at {src}")
    return pd.read_csv(src, index_col=[0])


if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    fire.Fire(write_cleaned_data)
