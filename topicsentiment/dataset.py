import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import pickle


class PandasDataset:
    """Class to simplify pre processing steps on dataframe. Requires prioir understanding of the dataset

    """
    def __init__(self):
        self.original_df = None
        self.current_df = None
        self.label_encoder = None

    def from_preprocessed(self, path: str):
        """Load from pre processed dataset file

        :param path: path to pre processed PandasDataset file
        :return:
        """
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
            self.original_df = dataset.original_df
            self.current_df = dataset.current_df
            self.label_encoder = dataset.label_encoder
            f.close()

    def read_data(self, filename: str = "sentisum-evaluation-dataset.csv"):
        """Load the CSV file into a workable format

        :param:
        :return: pd dataset
        """
        self.original_df = pd.read_csv(filename, header=None)
        data = self.original_df.fillna('')

        column_names = ['text']
        label_names = []
        for idx in range(1, 15):
            name = 'label_' + str(idx)
            label_names.append(name)
            column_names.append(name)

        data.columns = column_names

        data['topics'] = data[label_names].values.tolist()

        out_data = data[['text', 'topics']]

        def clean_topics(x):
            return [top for top in x if top != '']

        out_data['topics'] = out_data['topics'].map(clean_topics)

        self.current_df = out_data

        return self.current_df

    def replace_labels(self, label: str, target: str):
        """Replace occurances of all labels with the target label

        :param label: source label
        :param target: target label
        :return:
        """

        def replace_lab(x):
            return [top if top != label else target for top in x]

        self.current_df.topics = self.current_df.topics.map(replace_lab)

    def merge_labels(self, minimum_samples: int = 100, minority_label: str = 'others'):
        """Merge Labels with less than minimum samples

        :param minimum_samples:
        :param minority_label: name for the common label
        :return:
        """
        label_counts = self.current_df.topics.explode().value_counts()
        label_names = label_counts.index

        label_others = []
        for idx, label in enumerate(label_names):
            if label_counts[idx] < minimum_samples:
                label_others.append(label)

        def replace_others(x):
            new_labels = []
            for top in x:
                sent = top.split(' ')[-1]
                if top in label_others:
                    new_labels.append(' '.join([minority_label, sent]))
                else:
                    new_labels.append(top)
            return new_labels

        self.current_df.topics = self.current_df.topics.map(replace_others)

        return self.current_df

    def undersample_label(self, topic: str, fraction: float):
        """Undersample a given label. Selectively works on single occurances

        :param topic:
        :param fraction: fraction to retain
        :return:
        """
        temp_df = self.current_df[self.current_df.topics.apply(lambda x: topic in x)]
        temp_df = temp_df[temp_df.topics.str.len() == 1].sample(frac=fraction)

        single_label_data = self.current_df[self.current_df.topics.str.len() == 1]
        drop_index = single_label_data[single_label_data.topics.apply(lambda x: topic in x)].index
        self.current_df = self.current_df.drop(drop_index)
        self.current_df = self.current_df.append(temp_df)

    def undersample_label_combo(self, topic_a: str, topic_b: str, fraction: float):
        """Under sample a given combination of labels.
        todo Add a combo with more than 2 topics
        :param topic_a:
        :param topic_b:
        :param fraction: fraction to retain
        :return:
        """
        temp_df = self.current_df[self.current_df.topics.apply(lambda x: x == [topic_a, topic_b])]
        temp_df = temp_df[temp_df.topics.str.len() == 2].sample(frac=fraction)

        double_label_data = self.current_df[self.current_df.topics.str.len() == 2]
        drop_index = double_label_data[double_label_data.topics.apply(lambda x: x == [topic_a, topic_b])].index

        self.current_df = self.current_df.drop(drop_index)
        self.current_df = self.current_df.append(temp_df)

    def overview(self):
        """Gives a quick overview of the current dataframe

        :return:
        """
        return {
            "value_counts": self.current_df.topics.explode().value_counts(),
            "labels": self.current_df.topics.explode().unique(),
            "mean no. of tokens": self.current_df.text.str.split().str.len().std(),
            "mean no. of sentences": self.current_df.text.str.split('.').str.len().std()
        }

    def encode_labels(self):
        """Encode the label classes for classification using MultiLabelBinarizer

        :return: class list
        """
        def l2t(x):
            return tuple(x)

        self.current_df.topics = self.current_df.topics.map(l2t)

        self.label_encoder = MultiLabelBinarizer()

        self.current_df['encoded'] = self.label_encoder.fit_transform(self.current_df.topics.tolist()).tolist()

        return self.label_encoder.classes_.tolist()

    def train_test_split(self, test_size: float = 0.2):
        """Generate train and test sets

        :param test_size: test set fraction
        :return: train_dataset, test_dataset
        """
        train_dataset, test_dataset = train_test_split(self.current_df, test_size=test_size)
        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)
        return train_dataset, test_dataset

    def save_dataset(self, path: str):
        """

        :param path:
        :return:
        """
        output = open(path, 'wb')
        pickle.dump(self, output)
        output.close()


class TorchDataset(Dataset):
    """Cstom dataset for converting examples to features in Bert format

    """

    def __init__(self, dataframe, tokenizer, max_len):
        """

        :param dataframe: pandas dataframe
        :param tokenizer: bert tokenizer
        :param max_len: tokenizer max len
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.encoded
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        """Convert examples to features in Bert format

        :param index: idx
        :return:
        """
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.data['encoded'][index], dtype=torch.long)
        }

    def get_dataloader(self, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0):
        """Generate dataloaders for Pytorch training

        :param batch_size: batch size
        :param shuffle:
        :param num_workers:
        :return: dataloader
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
