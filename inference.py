import fire
import lxml.html
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def inference(
    text: str,
    path: str = "models/distilbert-base-uncased/run_5/checkpoint-2304/",
    name: str = "distilbert-base-uncased",
):
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    out = model(**tokenizer([text], return_tensors="pt"))
    print(out)


def get_html_content(url: str, max_size: int):
    try:
        str_content = download_limited(url, max_size)
    except Exception as e:
        return e
    return parse_content(str_content)


def download_limited(url, max_size) -> str:
    """Sends a request and limits the size of the response, while making sure
    the status code is 2xx."""
    res = requests.get(url, stream=True, timeout=5)

    code = res.status_code

    if code > 300 and code < 200:
        raise Exception(f"received status code {code}")

    if int(res.headers.get("Content-Length", 0)) > max_size:
        raise Exception(f"response size went over {max_size} limit")

    data = []
    length = 0

    for chunk in res.iter_content(1024):
        data.append(chunk)
        length += len(chunk)
        if length > max_size:
            raise Exception(f"response size went over {max_size} limit")

    return "".join(data)


def parse_content(text: str):
    """Parses the response html content and returns only text"""
    page = lxml.html.document_fromstring(text)
    page.cssselect("body")[0].text_content()


if __name__ == "__main__":
    fire.Fire(inference)
