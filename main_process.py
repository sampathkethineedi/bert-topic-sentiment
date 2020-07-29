from dataset import PandasDataset, TorchDataset
from transformers import BertTokenizer, BertConfig
import config
from model import BERTClass, Trainer, FocalLossLogits
import pandas as pd
import torch
import argparse

config = config.Settings()


def pre_process(verbose: bool = False):
    pd_dataset = PandasDataset("sentisum-evaluation-dataset.csv")
    pd_dataset.load_data()

    pd.set_option('display.max_rows', 100)
    pd_dataset.overview()
    if verbose:
        print(pd_dataset.overview())

    pd_dataset.adjust_labels()

    pd_dataset.drop_majority_labels('value for money positive', 0.1)
    pd_dataset.drop_majority_labels('garage service positive', 0.2)
    pd_dataset.drop_majority_label_combo('value for money positive', 'garage service positive', 0.2)

    pd_dataset.current_df.sample(frac=0.01)
    if verbose:
        print(pd_dataset.overview())

    pd_dataset.encode_labels()

    train_dataset, test_dataset = pd_dataset.train_test_split(0.2)

    return train_dataset, test_dataset


def prepare_train(train_dataset, test_dataset, verbose=False):
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL)

    torch_training_set = TorchDataset(train_dataset, tokenizer, config.MAX_LEN)
    torch_testing_set = TorchDataset(test_dataset, tokenizer, config.MAX_LEN)

    training_dataloader = torch_training_set.get_dataloader(batch_size=16)
    testing_dataloader = torch_testing_set.get_dataloader(batch_size=8)

    model_config = BertConfig()
    model = BERTClass(model_config)
    model.to(config.DEVICE)
    if verbose:
        print(model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    def loss_fn(outputs, targets):
        return FocalLossLogits()(outputs, targets)

    trainer = Trainer(config.MODEL_DIR)
    trainer.prepare(model, optimizer, loss_fn)

    if verbose:
        print("\n Trainer prepared")

    return trainer, training_dataloader, testing_dataloader


def train(trainer, training_dataloader,  testing_dataloader):
    trainer.train(config.EPOCHS, training_dataloader, testing_dataloader, validate=True)

# trainer.save_all(config.MODEL_DIR, tokenizer, torch_training_set.label_encoder)


def parse_args():
    """
    Use default arguments or add extra arguments
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train", dest="train", action="store_true", default=False)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dataset, test_dataset = pre_process(args.verbose)
    if args.train:
        prepare_train(train_dataset, test_dataset, args.verbose)
