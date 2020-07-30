# sentisum-topic-sentiment - dev
Topic Based Sentiment Detection using BERT

![alt text](https://github.com/sampathkethineedi/sentisum-topic-sentiment/blob/dev/st_interface.png?raw=true)

### Requirements
- transformers
- torch
- streamlit
- fastapi
- Pandas
- Alternatively, run the docker image with standard config

### Setup
- Clone
- Install requirements

## Train
Run ``

## Prediction API
Using FastAPI

Run `uvicorn prediction_api:app`

###  API Documentation
Swagger Docs at `http://127.0.0.1:8000/docs`

## Streamlit Interface
Run `streamlit run st_app.py`

App at `http://localhost:8501/`

## Planned Improvements
- Selective topic merging
- Trainer CLI
- Add visual metrics to Trainer
- Add colab notebook for data exploration
- Docker image

## References
- Add Alternative approaches
- Links to useful resources

