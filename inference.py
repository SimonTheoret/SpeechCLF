import fire
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import tracemalloc

def inference(
    path: str = "models/distilbert-base-uncased/run_5/checkpoint-2304/",
    name: str = "distilbert-base-uncased",
):
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    out = model(**tokenizer(["you a hoe bitch"], return_tensors="pt"))
    print(out)


if __name__ == "__main__":
    tracemalloc.start()
    fire.Fire(inference)
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()

