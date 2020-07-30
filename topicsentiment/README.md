
## Model Class
```
class BertForMultiLabel(BertPreTrainedModel):
    """Bert model with custom classifier for Multi Label classification
    Inherits from BertModel in transformers library

    """
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

## Trainer Class

```
class Trainer:
    """Trainer class to manage the complete training process

    """
    def __init__(self, model, optimizer, loss_fun):

    @staticmethod
    def setup_folders():

    def train(self, epochs: int, training_loader, testing_loader=None, validate: bool = False):
        """ Trains the model for input epochs

        :param epochs:
        :param training_loader:
        :param testing_loader:
        :param validate: if set to true Runs validation after every epoch
        :return: metrics dictionary
        """

    def validate(self, testing_loader):
        """ Runs validation on the model current state

        :param testing_loader:
        :return: metrics
        """

    def save_model(self, path: str):
        """Save only the model

        :param path:
        :return:
        """

    def save_all(self, path: str, tokenizer: BertTokenizer, label_encoder):
        """Save all files needed for inference

        :param path:
        :param tokenizer: Bert tokenizer
        :param label_encoder: label encoder
        :return:
        """

    def plot_metrics(self, metric: str, view: bool = False):
        """Plot an individual metric

        :param metric:
        :param view: if set to true opens a window
        :return:
        """

    def save_metric_plots(self):
        """Save all metric plots to model dir
        
        :return: 
        """
```
