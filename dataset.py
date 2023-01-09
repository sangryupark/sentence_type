import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, label):
        self.utterance = data["문장"]
        self.target = data[label].tolist()
        self.tokenized_sentence = tokenizer(
            self.utterance.tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=600,
            add_special_tokens=True,
        )

    def __getitem__(self, idx):
        encoded = {
            key: val[idx].clone().detach()
            for key, val in self.tokenized_sentence.items()
        }
        encoded["label"] = torch.tensor(self.target[idx])
        return encoded

    def __len__(self):
        return len(self.target)


class MultiDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.sentence = data["문장"].tolist()
        self.type = data["유형"].tolist()
        self.polarity = data["극성"].tolist()
        self.tense = data["시제"].tolist()
        self.certainty = data["확실성"].tolist()
        self.tokenized_sentence = tokenizer(
            self.sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=600,
            add_special_tokens=True,
        )

    def __getitem__(self, idx):
        encoded = {
            key: val[idx].clone().detach()
            for key, val in self.tokenized_sentence.items()
        }
        encoded["label"] = (
            torch.tensor(self.type[idx]),
            torch.tensor(self.polarity[idx]),
            torch.tensor(self.tense[idx]),
            torch.tensor(self.certainty[idx]),
        )
        return encoded

    def __len__(self):
        return len(self.type)
