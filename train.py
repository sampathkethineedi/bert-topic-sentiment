from topicsentiment.dataset import PandasDataset, TorchDataset
from transformers import BertTokenizer, BertConfig
import config
from topicsentiment.model import BertForMultiLabel, Trainer, FocalLossLogits
import pandas as pd
import torch
import argparse
import logging

config = config.Settings()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pre_process(filename: str = 'sentisum-evaluation-dataset.csv', verbose: bool = False):
    pd_dataset = PandasDataset(filename)
    pd_dataset.load_data()
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

    pd_dataset.encode_labels()

    train_dataset, test_dataset = pd_dataset.train_test_split(0.2)
    logger.info('Pre processing complete...')

    return train_dataset, test_dataset, pd_dataset.label_encoder


def prepare_train(train_dataset, test_dataset, verbose=False):
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL)

    torch_training_set = TorchDataset(train_dataset, tokenizer, config.MAX_LEN)
    torch_testing_set = TorchDataset(test_dataset, tokenizer, config.MAX_LEN)

    training_dataloader = torch_training_set.get_dataloader(batch_size=config.TRAIN_BATCH_SIZE)
    testing_dataloader = torch_testing_set.get_dataloader(batch_size=config.VALID_BATCH_SIZE)

    model_config = BertConfig()
    model = BertForMultiLabel(model_config)
    model.to(config.DEVICE)
    if verbose:
        print(model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    if config.LOSS_FUN == 'focal':
        def loss_fn(outputs, targets):
            return FocalLossLogits()(outputs, targets)

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
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dataset, test_dataset, label_encoder = pre_process(args.data, args.verbose)

    if args.train:
        trainer, training_dataloader, testing_dataloader, tokenizer = prepare_train(train_dataset,
                                                                                    test_dataset,
                                                                                    args.verbose)
        logger.info("Training started...")
        trainer.train(config.EPOCHS, training_dataloader, testing_dataloader, validate=True)
        trainer.save_metric_plots()
        logger.info("Training complete")

        trainer.save_all(config.MODEL_DIR, tokenizer, label_encoder)
        logger.info("Files saved at "+config.MODEL_DIR)
