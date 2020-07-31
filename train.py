from topicsentiment.dataset import PandasDataset, TorchDataset
from transformers import BertTokenizer, BertConfig
import config
from topicsentiment.model import BertForMultiLabel, Trainer, FocalLossLogits
import pandas as pd
import torch
import argparse
import logging
import os
import sys

config = config.Settings()


# Setting up logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(config.MODEL_DIR, 'training.log'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def pre_process(filename: str = 'sentisum-evaluation-dataset.csv', verbose: bool = False):
    """ Pre Processing. Takes in the CSV file and returns a 'PandasDataset' object

    :param filename: CSV file name
    :param verbose: To show Dataset overveiw after each step
    :return:
    """
    # Create a PandasDataset
    pd_dataset = PandasDataset()
    pd_dataset.read_data(filename)
    logger.info('Dataset loaded')

    pd.set_option('display.max_rows', 100)
    if verbose:
        print(pd_dataset.overview())

    # Fixing dataset specific issues
    # Method can be sed to merge similar labels with low samples
    pd_dataset.replace_labels('advisor/agent service negative', 'advisoragent service negative')
    pd_dataset.replace_labels('advisor/agent service positive', 'advisoragent service positive')
    logger.info('Replacing labels complete')

    pd_dataset.merge_labels(minimum_samples=100, minority_label="others")
    logger.info('Minority labels adjusted')

    # Under sampling majority labels
    pd_dataset.undersample_label('value for money positive', 0.1)
    pd_dataset.undersample_label('garage service positive', 0.2)
    pd_dataset.undersample_label_combo('value for money positive', 'garage service positive', 0.2)
    logger.info('Undersampling complete')

    if verbose:
        print(pd_dataset.overview())

    # Encode the topics for multi label classification
    pd_dataset.encode_labels()

    return pd_dataset


def prepare_train(pd_dataset: PandasDataset, verbose=False):
    """Prepares the Trainer class and the dependencies. Returns the trainer

    :param pd_dataset: PandasDataset object
    :param verbose:
    :return:
    """

    # Create train and test splits
    train_dataset, test_dataset = pd_dataset.train_test_split(0.2)
    logger.info('Test Split complete...')

    # Initialise BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL)

    # Create TorchDatasets for train and test
    torch_training_set = TorchDataset(train_dataset, tokenizer, config.MAX_LEN)
    torch_testing_set = TorchDataset(test_dataset, tokenizer, config.MAX_LEN)

    # Create DataLoaders for train and test to be passed in to the model
    training_dataloader = torch_training_set.get_dataloader(batch_size=config.TRAIN_BATCH_SIZE)
    testing_dataloader = torch_testing_set.get_dataloader(batch_size=config.VALID_BATCH_SIZE)

    # Initialise the model
    model_config = BertConfig()
    model = BertForMultiLabel(model_config)
    model.to(config.DEVICE)
    if verbose:
        print(model)

    # Optimizer to be passed into Trainer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    # Loss Function to be passed into Trainer. Defaults to BCE
    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    if config.LOSS_FUN == 'focal':
        def loss_fn(outputs, targets):
            return FocalLossLogits()(outputs, targets)

    # Initialise Trainer
    trainer = Trainer(model, optimizer, loss_fn)

    logger.info("Trainer prepared")

    return trainer, training_dataloader, testing_dataloader, tokenizer


def parse_args():
    """Argument parser

    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train", dest="train", action="store_true", default=False)
    parser.add_argument("--preprocess", dest="preprocess", action="store_true", default=False)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)

    return parser.parse_args()


def train(pd_dataset, verbose):
    """Main Training process

    :param pd_dataset: Pandas Dataset
    :param verbose: verbose flag
    :return:
    """

    try:
        # Prepare the Trainer
        trainer, training_dataloader, testing_dataloader, tokenizer = prepare_train(pd_dataset, verbose)
        logger.info("Training started...")
    except Exception as err:
        logger.error(err)
        sys.exit()

    try:
        # Train and save metric plots
        trainer.train(config.EPOCHS, training_dataloader, testing_dataloader, validate=True)
        trainer.save_metric_plots()
        logger.info("Training complete")
    except Exception as err:
        logger.error(err)
        sys.exit()

    try:
        # Save all files needed for inference
        trainer.save_all(config.MODEL_DIR, tokenizer, pd_dataset.label_encoder)
        logger.info("Files saved at " + config.MODEL_DIR)
    except Exception as err:
        logger.error(err)
        sys.exit()


if __name__ == "__main__":
    args = parse_args()

    if args.preprocess:
        try:
            pd_dataset = pre_process(args.data, args.verbose)
            pd_dataset.save_dataset(os.path.join(config.MODEL_DIR, 'final_data.pkl'))
        except Exception as err:
            logger.error(err)
            sys.exit()

        logger.info("Cleaned dataset saved at "+os.path.join(config.MODEL_DIR, 'final_data.pkl'))

    elif args.train:
        pd_dataset = PandasDataset()
        pd_dataset.from_preprocessed(args.data)
        train(pd_dataset, args.verbose)

    else:
        pd_dataset = pre_process(args.data, args.verbose)
        train(pd_dataset, args.verbose)


