# sentisum-topic-sentiment - dev
Topic Based Sentiment Detection using BERT

![alt text](https://github.com/sampathkethineedi/sentisum-topic-sentiment/blob/dev/st_interface.png?raw=true)

### Requirements
- transformers
- torch
- streamlit
- fastapi
- pandas

### Setup
- Clone the repo
- Install requirements
- Download pre-trained model files here

## Training

Pandas and Torch dataset classes in **topicsentiment/dataset.py**

Model Class and Trainer Class in **topicsentiment/model.py**

Jump to topicsentiment for detailed info

Configuration in **config.py**

Run `python main_process.py --data sentisum-evaluation-dataset.csv --train`

## Prediction API
Built using FastAPI

Run `uvicorn prediction_api:app`

Swagger Docs at `http://127.0.0.1:8000/docs`

## Streamlit Interface
Simple Interface built using streamlit

Code in **st_app.py**

Run `streamlit run st_app.py`

App at `http://localhost:8501/`

## References
- Add Alternative approaches
- Links to useful resources

