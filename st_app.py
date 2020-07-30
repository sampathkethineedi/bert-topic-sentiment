import streamlit as st
import requests
import json

SPAWNER_API = "http://127.0.0.1:8000/"
headers = {'content-type': 'application/json'}


st.write("## Topic Based Sentiment Detection")
st.write("### Fine Tuning BERT for multi label classification")
st.write("API Docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")

input_text: str = st.text_area(label="Input Text")

st.sidebar.text("Params")
threshold: float = st.sidebar.slider(label="threshold", min_value=0.2, max_value=0.8)
max_len: float = st.sidebar.slider(label="tokenizer max length", min_value=32, max_value=128)


if st.button("Submit"):
    body = json.dumps({"input_text": input_text, "threshold": threshold})
    response = requests.post(url=SPAWNER_API+'predict', headers=headers, data=body).json()
    for pred in response["prediction"]:
        output = ' '.join(pred.split(' ')[:-1])
        if 'positive' in pred:
            st.success(output)
        else:
            st.error(output)

st.write("**Developer**: Sampath Kethineedi")
st.write("**Mail**: sampath.kethineedi9@gmail.com")
st.write("Code at my GitHub repo [bert-topic-sentiment](https://github.com/sampathkethineedi/bert-topic-sentiment.git)")
st.write("**Dataset by SentiSum**")
