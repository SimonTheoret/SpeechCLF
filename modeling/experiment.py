import os
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv

import wandb


@dataclass
class ModelHyperparamters(dict):
    """Class holding hyperparameters. Inherits from the dict class."""

    n_layers = None
    n_heads = None
    dim = None
    hidden_dim = None
    dropout = None
    attention_dropout = None
    activation = None
    dataset: str = "data/cleaned.csv"

    def to_dict(self) -> dict[str, Any]:
        """Returns the hyperparameters as a dict and removes `None` values"""
        di = self.__dict__
        for k, v in di.items():
            if v is None:
                di.pop(k)
        return di


@dataclass
class TrainingHyperparameters(dict):
    """Class holding hyperparameters. Inherits from the dict class."""

    do_train: bool
    do_eval: bool
    learning_rate = 5e-5
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    evaluation_strategy: str = "epoch"
    weight_decay: Optional[float] = 0
    lr_scheduler_type: Optional[str] = "linear"
    output_dir: Optional[str] = None
    save_strategy: str = "epoch"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Returns the hyperparameters as a dict"""
        return self.__dict__

    def set_train(self, train_dataset):
        self.train_dataset = train_dataset

    def set_eval(self, eval_dataset):
        self.eval_dataset = eval_dataset


def init_exp(
    thp: TrainingHyperparameters,
    mhp: ModelHyperparamters,
    tags: list[str],
    name: str,
    project: str = "SpeechCLF",
):
    """Initiate an experimentation on wandb.
    Parameters
    ----------
    hp: Hyperparameters
        The hyperparameters for the experiment
    tags: list[str]
        A list of tags for the experiment
    name: str
        Name given to the run
    project: str
        Project for this experiment. Defaults to `SpeechCLF`.
    """
    load_dotenv(".env")
    key = os.getenv("API_KEY")
    wandb.login(key=key, verify=True)

    wandb.init(
        project=project,
        tags=tags,
        name=name,
        config={
            "trainer_params": thp.to_dict(),
            "model_params": mhp.to_dict(),
        },  # pyright: ignore
    )


def log_results(metrics: dict[str, Any]):
    """Logs the results to wandb.
    Parameters
    ----------
    metrics: dict[str, Any]
        The metrics to log into wandb.
    """
    wandb.log(metrics)
    print("logged metrics: ", metrics)
