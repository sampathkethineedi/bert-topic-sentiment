## Model Class
```
class BertForMultiLabel(BertPreTrainedModel):
    def __init__(self, model_config):
        super(BertForMultiLabel, self).__init__(model_config)
        self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL)
        self.drp = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, config.NUM_LABELS)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.drp(output_1)
        output = self.classifier(output_2)
        return output
```

