import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from argument import TrainModelArguments
from dataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from utils import num_to_label


def inference():
    dataset = pd.read_csv("./data/test.csv")
    parser = HfArgumentParser(TrainModelArguments)
    (model_args,) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    target = ["유형", "극성", "시제", "확실성"]

    if model_args.k_fold:
        print("### START INFERENCE with KFold ###")
        for idx, t in enumerate(target):
            print(f"Start inference {t}")
            dataset[t] = [-1] * len(dataset)
            test_dataset = CustomDataset(dataset, tokenizer, t)
            dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            pred_prob = []
            for fold_num in range(1, model_args.fold_num + 1):
                print(f"--- START INFERENCE FOLD {fold_num} ---")
                model_path = os.path.join(
                    "./output/",
                    model_args.project_name + "_kfold",
                    str(idx),
                    "fold" + str(fold_num),
                )
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(device)
                model.eval()

                output_prob = []
                for data in tqdm(dataloader):
                    with torch.no_grad():
                        outputs = model(
                            input_ids=data["input_ids"].to(device),
                            attention_mask=data["attention_mask"].to(device),
                            token_type_ids=data["token_type_ids"].to(device),
                        )
                        logit = outputs[0]
                        prob = F.softmax(logit, dim=-1).detach().cpu().numpy()
                        output_prob.append(prob)
                output_prob = np.concatenate(output_prob, axis=0).tolist()
                pred_prob.append(output_prob)
                print(f"--- FINISH INFERENCE FOLD {fold_num} ---")

            pred_prob = np.sum(pred_prob, axis=0) / model_args.fold_num
            pred_answer = np.argmax(pred_prob, axis=-1)
            dataset[t] = pred_answer
            dataset[t] = num_to_label(dataset[t], t)

    else:
        print("### START INFERENCE with Non-KFold ###")
        for idx, t in enumerate(target):
            print(f"Start inference {t}")
            dataset[t] = [-1] * len(dataset)
            test_dataset = CustomDataset(dataset, tokenizer, t)
            dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            model_path = os.path.join("./output/", model_args.project_name, str(idx))
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()

            output_prob = []
            output_pred = []

            for data in tqdm(dataloader):
                output = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    token_type_ids=data["token_type_ids"].to(device),
                )
                logit = output[0]
                prob = F.softmax(logit, dim=-1).detach().cpu().numpy()
                logit = logit.detach().cpu().numpy()
                result = np.argmax(logit, axis=-1)
                output_pred.append(result)
                output_prob.append(prob)

            pred_answer = np.concatenate(output_pred).tolist()
            output_prob = np.concatenate(output_prob, axis=0).tolist()
            dataset[t] = pred_answer
            dataset[t] = num_to_label(dataset[t], t)
            print(f"Finish inference {t}")

    return_answer = []
    for idx in tqdm(range(len(dataset))):
        answer = (
            dataset["유형"][idx]
            + "-"
            + dataset["극성"][idx]
            + "-"
            + dataset["시제"][idx]
            + "-"
            + dataset["확실성"][idx]
        )
        return_answer.append(answer)

    submission = pd.DataFrame({"ID": dataset["ID"], "label": return_answer})
    if model_args.k_fold:
        submission.to_csv(
            os.path.join(
                "./output", model_args.project_name + "_kfold", "submission.csv"
            ),
            index=False,
        )
    else:
        submission.to_csv(
            os.path.join("./output/", model_args.project_name, "submission.csv"),
            index=False,
        )
    print("### INFERENCE FINISH ###")


if __name__ == "__main__":
    inference()
