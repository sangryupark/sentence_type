import os
import pandas as pd
import torch
import wandb

from argument import TrainingArguments, TrainModelArguments
from dataset import CustomDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from trainer import CustomTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from utils import label_to_num


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

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    set_seed(train_args.seed)

    target = ["유형", "극성", "시제", "확실성"]
    target_num = [4, 3, 3, 2]

    print("### START TRAINING with Non-KFold ###")
    for idx, t in enumerate(target):
        print(f"Start Training {t}")
        data[t] = label_to_num(data[t], t)
        model_config = AutoConfig.from_preptrained(
            pretrained_model_name_or_path=model_args.model_name
        )
        model_config.num_labels = target_num[idx]
        model = AutoModelForSequenceClassification(
            pretrained_model_name_or_path=model_args.model_name, config=model_config
        )
        model.to(device)
        model.train()

        wandb.init(
            entity="psrpsj",
            project="sentence",
            name=model_args.project_name + "_" + t,
            tags=[model_args.model_name],
        )
        wandb.config.update(train_args)

        train_dataset, valid_dataset = train_test_split(
            data, test_size=0.2, stratify=data[t], random_state=42
        )

        train = CustomDataset(train_dataset, tokenizer)
        valid = CustomDataset(valid_dataset, tokenizer)

        trainer = CustomTrainer(
            model=model,
            args=train_args,
            loss_name=model_args.loss_name,
            train_data=train,
            eval_data=valid,
        )
        model.save_pretrained(
            os.path.join(train_args.output_dir, model_args.project_name, str(idx))
        )
        wandb.finish()
        print(f"Training {t} finish!")
    print("### TRAINING FINISH ###")