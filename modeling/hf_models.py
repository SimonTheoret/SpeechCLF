import json
import os
import pathlib
from dataclasses import dataclass

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from modeling.experiment import ModelHyperparamters, TrainingHyperparameters
from preprocessing.data_processing import split_dataset

# Model pipeline
# select model name -> verify GPU ready -> get model tokenizer -> tokenize data ->
# select model's HP/config -> fine tune model -> start wandb experiment -> test model ->
# save model weights  -> save exp results on wannb -> Done


@dataclass
class HfPipeline:
    """Generic class to build an train NLP models from HuggingFace. It only need the
    `model_name` to be initialized."""

    model_name: str
    thp: TrainingHyperparameters
    # Training arguments given directly to the trainer as TrainingConfig
    mhp: ModelHyperparamters

    def set_gpu_state(self, testing: bool):
        if testing:
            self.device = "cpu"
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_batch_len(
        self,
        set_padding_token: bool = False,
        padding: str | bool = True,
        truncation: str | bool = True,
    ):
        self.set_padding_token: bool = set_padding_token
        self.add_padding: str | bool = padding
        self.truncation = truncation

    def set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_len=512)
        if self.set_padding_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def set_model_config(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_name, **self.mhp.to_dict(), num_labels=3
        )

    def set_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            config=self.model_config,
        ).to(self.device)

    def build_dataset(self, src: str = "./data/cleaned.csv"):
        train_df, valid_df, test_df = split_dataset(src)
        self.train_ds = Dataset.from_pandas(train_df)  # pyright: ignore
        self.eval_ds = Dataset.from_pandas(valid_df)  # pyright: ignore
        self.test_ds = Dataset.from_pandas(test_df)  # pyright: ignore
        self.mhp.dataset = src  # overrides the dataset in training HP

    def tokenize(self):
        self.train_ds = self.train_ds.map(
            lambda examples: self.tokenizer(
                examples["tweet"],
                padding=self.add_padding,
                return_tensors="np",
                truncation=self.truncation,
            ),
            batched=True,
        )
        self.eval_ds = self.eval_ds.map(
            lambda examples: self.tokenizer(
                examples["tweet"],
                padding=self.add_padding,
                return_tensors="np",
                truncation=self.truncation,
            ),
            batched=True,
        )
        self.test_ds = self.test_ds.map(
            lambda examples: self.tokenizer(
                examples["tweet"],
                padding=self.add_padding,
                return_tensors="np",
                truncation=self.truncation,
            ),
            batched=True,
        )

    # def set_tokenized_dataset(self):
    #     self.thp.set_train(self.train_ds)
    #     self.thp.set_eval(self.eval_ds)

    def setup_dir(self):
        default_dir = "models/" + self.model_name + "/run_"
        for i in range(1000):
            if os.path.exists(default_dir + str(i)):
                self.dir_destination = default_dir + str(i) + "/"
                self.thp.output_dir = self.dir_destination
                self.run_n = i
                continue
            else:
                pathlib.Path(default_dir + str(i)).mkdir(parents=True)
                self.dir_destination = default_dir + str(i) + "/"
                self.thp.output_dir = self.dir_destination
                self.run_n = i
                return

    def set_training_args(self, testing: bool):
        self.training_args = TrainingArguments(**self.thp.to_dict(), use_cpu=testing)
        print(self.training_args.device)

    def set_evaluation(self):
        self.evaluation_metric = evaluate.load("accuracy")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.evaluation_metric.compute(
            predictions=predictions, references=labels
        )

    def set_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            train_dataset=self.train_ds,  # pyright: ignore
            eval_dataset=self.eval_ds,  # pyright: ignore
            compute_metrics=self.compute_metrics,  # pyright: ignore
        )

    def write_results(self):
        with open(self.dir_destination + "train.txt", "w") as f1:
            f1.write(str(self.out_train))

        with open(self.dir_destination + "eval.json", "w") as f2:
            json.dump(self.out_eval, f2)

    def finetune(
        self,
        src: str = "data/cleaned.csv",
        testing: bool = False,
        padding: str | bool = True,
        set_padding_token: bool = False,
        truncation: str | bool = True,
    ):
        "Run everything"
        self.set_gpu_state(testing)
        self.set_model_config()
        self.set_model()
        self.set_batch_len(set_padding_token, padding, truncation)
        self.build_dataset(src)
        self.set_tokenizer()
        self.tokenize()
        self.setup_dir()
        # self.set_tokenized_dataset()
        self.set_training_args(testing)
        self.set_evaluation()
        self.set_trainer()
        self.out_train = self.trainer.train()
        self.out_eval = self.trainer.evaluate()
        self.write_results()
