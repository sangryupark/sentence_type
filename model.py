import torch.nn as nn

from transformers import AutoModel


class MultiLabelModel(nn.Module):
    def __init__(self, model, config):
        super(MultiLabelModel, self).__init__()
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model, config=config
        )

        self.type_classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features=768, out_features=4)
        )
        self.polarity_classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features=768, out_features=3)
        )
        self.tense_classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features=768, out_features=3)
        )
        self.certainty_classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features=768, out_features=2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        type_output = self.type_classifier(output[:, 0, :].view(-1, 768))
        polarity_output = self.polarity_classifier(output[:, 0, :].view(-1, 768))
        tense_output = self.tense_classifier(output[:, 0, :].view(-1, 768))
        certainty_output = self.certainty_classifier(output[:, 0, :].view(-1, 768))
        return type_output, polarity_output, tense_output, certainty_output
