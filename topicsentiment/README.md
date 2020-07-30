
## BERT Model Class

```python
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

## Main Trainer Class

```python
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

## Pandas Dataset

```python
class PandasDataset:
    """Class to simplify pre processing steps on dataframe. Requires prioir understanding of the dataset

    """
    def __init__(self, filename: str = "sentisum-evaluation-dataset.csv"):

    def load_data(self):
        """Load the CSV file into a workable format

        :param:
        :return: pd dataset
        """

    def replace_labels(self, label: str, target: str):
        """Replace occurances of all labels with the target label

        :param label: source label
        :param target: target label
        :return:
        """

    def merge_labels(self, minimum_samples: int = 100, minority_label: str = 'others'):
        """Merge Labels with less than minimum samples

        :param minimum_samples:
        :param minority_label: name for the common label
        :return:
        """

    def undersample_label(self, topic: str, fraction: float):
        """Undersample a given label. Selectively works on single occurances

        :param topic:
        :param fraction: fraction to retain
        :return:
        """

    def undersample_label_combo(self, topic_a: str, topic_b: str, fraction: float):
        """Under sample a given combination of labels.
        todo Add a combo with more than 2 topics
        :param topic_a:
        :param topic_b:
        :param fraction: fraction to retain
        :return:
        """

    def overview(self):
        """Gives a quick overview of the current dataframe

        :return:
        """

    def encode_labels(self):
        """Encode the label classes for classification using MultiLabelBinarizer

        :return: class list
        """

    def train_test_split(self, test_size: float = 0.2):
        """Generate train and test sets

        :param test_size: test set fraction
        :return: train_dataset, test_dataset
        """
```

## Torch Datset

```python
class TorchDataset(Dataset):
    """Cstom dataset for converting examples to features in Bert format

    """

    def __init__(self, dataframe, tokenizer, max_len):
        """

        :param dataframe: pandas dataframe
        :param tokenizer: bert tokenizer
        :param max_len: tokenizer max len
        """

    def __len__(self):

    def __getitem__(self, index):
        """Convert examples to features in Bert format

        :param index: idx
        :return:
        """

    def get_dataloader(self, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0):
        """Generate dataloaders for Pytorch training

        :param batch_size: batch size
        :param shuffle:
        :param num_workers:
        :return: dataloader
        """
```
