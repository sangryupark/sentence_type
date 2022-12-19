import numpy as np
import torch
import torch.nn.functional as F

from loss import create_criterion
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from transformers import (
    Trainer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


class CustomTrainer(Trainer):
    """Custom Loss를 적용하기 위한 Trainer"""

    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name

    def compute_loss(self, model, inputs, return_outputs=False):

        if "labels" in inputs and self.loss_name != "default":
            custom_loss = create_criterion(self.loss_name)
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            loss = custom_loss(outputs[0], labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
