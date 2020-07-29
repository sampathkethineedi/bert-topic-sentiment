from pydantic import BaseSettings


class Settings(BaseSettings):

    NUM_LABELS: int = 24
    MAX_LEN: int = 64
    TRAIN_BATCH_SIZE: int = 8
    VALID_BATCH_SIZE: int = 4
    EPOCHS: int = 1
    LEARNING_RATE: int = 1e-05

    PRE_TRAINED_MODEL: str = "bert-base-uncased"
    DEVICE: str = "cpu"

    MODEL_DIR: str = "model_files"
    MODEL_NAME: str = "bert-topic-sentiment.bin"
