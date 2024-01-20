import fire

from modeling.experiment import ModelHyperparamters, TrainingHyperparameters, init_exp
from modeling.hf_models import HfPipeline


def set_thp(kwargs) -> TrainingHyperparameters:
    print(f"Training hyperparmeters given: { kwargs }")
    return TrainingHyperparameters(True, True, **kwargs)


def set_mhp(kwargs) -> ModelHyperparamters:
    print(f"Model hyperparmeters given: { kwargs }")
    return ModelHyperparamters(**kwargs)


def run(
    name,
    tags_list,
    model_name: str,
    thpkwargs={},
    mhpkwargs={},
    set_padding_token=False,
    padding="max_length",
    truncation="longest_first",
    src: str = "data/cleaned.csv",
):
    thp = set_thp(thpkwargs)
    mhp = set_mhp(mhpkwargs)
    print(f"Model name: {model_name}")
    pipeline = HfPipeline(model_name, thp, mhp)
    init_exp(thp, mhp, tags_list, name)
    print(f"Data source: {src}")
    pipeline.finetune(
        src=src,
        set_padding_token=set_padding_token,
        padding=padding,
        truncation=truncation,
    )


if __name__ == "__main__":
    fire.Fire(run)
