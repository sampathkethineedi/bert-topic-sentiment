from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import BertConfig, AutoTokenizer
from topicsentiment.model import BertForMultiLabel
import torch
import config
from typing import List
import pickle
import numpy as np
import os

config = config.Settings()

# Load the model fro inference
model_config = BertConfig()
model = BertForMultiLabel(model_config)
model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, config.MODEL_NAME_COLAB), map_location=config.DEVICE))
tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL)

with open(os.path.join(config.MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
    f.close()


app = FastAPI(title='SentiSum Topic Based Sentiment Prediction API',
              description='Fine Tuning multi label classification model on top of BERT',
              version="1.0")


class PredictInput(BaseModel):
    """Input model for prediction

    """
    input_text: str = Field(None, description='Input Text', example="Excellent price, good choice of garages and "
                                                                    "trouble free prompt fitting, with no wait")
    threshold: float = Field(0.3, description="Prediction Threshold", example=0.3)
    max_len: int = Field(64, description="Tokenizer max len", example=64)


class PredictResponse(BaseModel):
    prediction: List[str] = Field(None, description="prediction labels", example=["length of fitting positive",
                                                                                  "value for money positive",
                                                                                  "wait time positive"])


@app.get("/")
def home():
    return "Refer to '/docs' for API documentation"


@app.post("/predict", description="Predict", response_model=PredictResponse)
def get_prediction(req_body: PredictInput):
    """Prediction

    :param req_body:
    :return:
    """

    # Encode the text to features
    encoded_text = tokenizer.encode_plus(
        req_body.input_text,
        max_length=req_body.max_len,
        add_special_tokens=True,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
    )

    try:

        # Generate the output and pass through sigmoid
        output = model(encoded_text["input_ids"], encoded_text["attention_mask"],
                       encoded_text["token_type_ids"]).sigmoid()

        # Filter out predictions less than the threshold
        prediction = [1 if i > req_body.threshold else 0 for i in output[0]]

        label = label_encoder.inverse_transform(np.array([prediction]))[0]

        return {"prediction": label}
    except Exception as err:
        raise HTTPException(status_code=666, detail="Error during prediction")


@app.get("/labels", description="Labels")
def get_labels():
    """List out the label names

    :return:
    """

    return {
        "num_labels": len(label_encoder.classes_.tolist()),
        "label_names": label_encoder.classes_.tolist()
    }
