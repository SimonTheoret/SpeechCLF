from modeling.experiment import ModelHyperparamters, TrainingHyperparameters
from modeling.hf_models import HfPipeline


def test_tokenized_datasets():
    thp = TrainingHyperparameters(False, False)
    mhp = ModelHyperparamters()
    pipe = HfPipeline("distilbert-base-uncased", thp, mhp)
    pipe.set_batch_len()
    pipe.build_dataset()
    pipe.set_tokenizer()
    pipe.tokenize()
    for sample in pipe.train_ds:
        assert all(type(i) is int for i in sample["input_ids"])  # pyright: ignore
        assert sample["label"] in [0, 1, 2]  # pyright: ignore

    for sample in pipe.eval_ds:
        assert all(type(i) is int for i in sample["input_ids"])  # pyright: ignore
        assert sample["label"] in [0, 1, 2]  # pyright: ignore

    for sample in pipe.test_ds:
        assert all(type(i) is int for i in sample["input_ids"])  # pyright: ignore
        assert sample["label"] in [0, 1, 2]  # pyright: ignore


# def test_pipeline():
#     mhp = ModelHyperparamters()
#     thp = TrainingHyperparameters(
#         True,
#         False,
#     )
#     pipe = HfPipeline(
#         model_name=(
#             "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"
#         ),
#         mhp=mhp,
#         thp=thp,
#     )
#     init_exp(thp, mhp, ["testrun", "debugging"], "testrunfullpipeline")
#     pipe.finetune(src="data/fake_dataset.csv", testing=True)
