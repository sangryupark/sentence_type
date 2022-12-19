import os
import pandas as pd
import torch
import wandb

from argument import TrainingArguments, TrainModelArguments
from dataset import CustomDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from trainer import CustomLossTrainer, CustomTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from utils import label_to_num, stopword


def compute_metrics(pred):
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(label, preds)
    f1 = f1_score(label, preds, average="weighted")
    return {"accuracy": acc, "f1_score": f1}


def train():
    data = pd.read_csv("./data/train.csv")
    parser = HfArgumentParser((TrainingArguments, TrainModelArguments))
    (train_args, model_args) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Current model is {model_args.model_name}")
    print(f"Current device is {device}")
