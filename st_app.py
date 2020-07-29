import streamlit as st
import requests
import json

SPAWNER_API = "http://127.0.0.1:8000/"
headers = {'content-type': 'application/json'}


st.write("# SentiSum Topic Based Sentiment Detection")
st.write("### Fine Tuning multi label classification model on top of BERT")
input_text: str = st.text_area(label="input text")

st.sidebar.text("Params")
threshold: float = st.sidebar.slider(label="threshold", min_value=0.2, max_value=0.8)
max_len: float = st.sidebar.slider(label="max length", min_value=32, max_value=128)


if st.button("Submit"):
    body = json.dumps({"input_text": input_text, "threshold": threshold})
    response = requests.post(url=SPAWNER_API+'predict', headers=headers, data=body).json()
    st.write(response["prediction"])

