from typing import Any

import fire
import lxml.html
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing.data_processing import preprocess_str


def inference(
    text: str,
    path: str = "output/google/MobileBert",
    name: str = "google/mobilebert-uncased",
) -> Any:
    """Prints and returns the output of the selected model."""
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    out = model(**tokenizer([text], return_tensors="pt"))
    print(out)
    return out


def get_html_content(url: str, max_size: int, length: int) -> list[str]:
    try:
        str_content = download_limited(url, max_size)
    except Exception as e:
        raise e
    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words("english")
    str_content = preprocess_str(str_content, lemmatizer, stopwords_english, None)
    str_content = parse_content(str_content)
    return split_strings(str_content, length)


def download_limited(url, max_size) -> str:
    """Sends a request and limits the size of the response, while making sure
    the status code is 2xx."""
    res = requests.get(url, stream=True, timeout=5)

    code = res.status_code

    if code > 300 and code < 200:
        raise Exception(f"received status code {code}")

    if int(size := res.headers.get("Content-Length", 0)) > max_size:
        raise Exception(f"response size (of size {size}) went over {max_size} limit")

    data = []
    length = 0

    for chunk in res.iter_content(1024):
        data.append(chunk.decode())
        length += len(chunk)
        if length > max_size:
            raise Exception(f"response size {length} went over {max_size} bytes limit")
    print(f"size of response: { length }")
    return "".join(data)


def parse_content(text: str) -> str:
    """Parses the response html content and returns only the text content"""
    page = lxml.html.fromstring(text)
    return page.cssselect("body")[0].text_content()


def split_strings(text: str, length: int) -> list[str]:
    "Split string into a list of string of specified length"
    text_len = len(text)
    splitted = [text[i : i + length] for i in range(0, text_len, length)]
    # if text_len % length != 0:  # if length does not divide the length of the text
    #     remaining = text_len % length  # characters not in the splitted
    #     splitted.append(
    #         text[-remaining - 1 : -1]
    #     )  # include -remaining-1 as it is not included in splitted
    return splitted


if __name__ == "__main__":
    fire.Fire(inference)
