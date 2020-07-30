# Topic Based Sentiment Detection using BERT

![alt text](https://github.com/sampathkethineedi/sentisum-topic-sentiment/blob/dev/st_interface.png?raw=true)

### Requirements
- transformers
- torch
- streamlit
- fastapi
- pandas

### Setup
- Clone the repo
- Install requirements: `conda create --name topicsentiment --file requirements.txt` or `pip install -r requirements.txt` in your env
- Download pre-trained model files [here](https://drive.google.com/drive/folders/1wWui9xZk0fnPzV06OHaKBS8xqJSOLPzS?usp=sharing)
- Copy the files to *model_dir* in config

#### Running the demo
- Run `uvicorn prediction_api:app`
- Run `streamlit run st_app.py`
- Go to `http://localhost:8501/`

## Training

Pandas and Torch dataset classes in **topicsentiment/dataset.py**

Model Class and Trainer Class in **topicsentiment/model.py**

Jump to topicsentiment for detailed info

Configuration in **config.py**

Run `python train.py --data sentisum-evaluation-dataset.csv --train`

## Prediction API
Built using FastAPI

Code in **prediction_api.py**

Run `uvicorn prediction_api:app`

Swagger Docs at `http://127.0.0.1:8000/docs`

## Streamlit Interface
Simple Interface built using streamlit

Code in **st_app.py**

Run `streamlit run st_app.py`

App at `http://localhost:8501/`

## Planned improvements
- Alternative approaches
- Notebook for data exploration
- Docker image

## References
- Add Alternative approaches
- Links to useful resources

