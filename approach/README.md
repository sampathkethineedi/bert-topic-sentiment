## Improvements and Alternate approaches

- Even fine-tuning the Transformer requires significant compute resources. With the BERT base model I had to wait 10-15 mins on Colab GPU for each iteration. There are more powerful models like T5 or BERT large which could give better results.

- Experiment with more layers on top of BERT. My implementation only has a dropout and a linear layer.

- Experiment with Batch Sizes, Learning rates and optimizers. These usually have marginal improvemetns so I did not spend time on this

- Experiment with custom loss function. I used Binary Cross Entropy and Focal Loss and found BCE was better. I intentionally ignore the imbalance to let the model learn the frequency patterns.

- Cluster the topics and use a hierarchial classifier. Intuitive clusters - **service, monetory, time**. This might cover even a few topics I ignored due to low samples.

- Selective merging of topics based on co-occurance.

- I considered a **clause level** topic and sentiment classification but the dataset only has sample level values for topic. And majority of sentences do not follow standard english syntax so dependedncy parsing was not possible to get the clauses.

## Current Approach 

I have considered this as a multi-label classification problem and used transformer based solution to fine tune a pre-trained BERT model. 

### Primary reasons for this
- Relatively small Dataset
- Niche Domain
- Leverage the information in language model
- Online Reviews are not rich in structure
- Ease of implementation

### Evaluation Metrics

FULL Dataset: 7757 | TRAIN Dataset: 6205 | VAL Dataset: 1552

- Training Accuracy Saturated at around **0.8** after 7 Epochs

- Training Loss quickly reached **0.2** and saturated at **0.008** after 7 Epochs

| Metric | Score |
| --- | --- |
| Accuracy Score | 0.8200171821305842 |
| Hamming Score | 0.9051907142113328 |
| F1 Score (Micro) | 0.9182371701942591 |
| F1 Score (Macro) | 0.7874733783966089 |
| F1 Score (Weighted)| 0.9039791053690711 |

## Understanding the data

There are **62 unique topic-sentiment labels** (after fixing a few duplicates)

![alt text](https://github.com/sampathkethineedi/sentisum-topic-sentiment/blob/master/approach/labels_dist.png?raw=true)

Most of the topics have both positive and negative samples

There are a lot of labels with less than a 100 samples. I have categorised these into two labels - **others positive** and **others negative**

![alt text](https://github.com/sampathkethineedi/sentisum-topic-sentiment/blob/master/approach/labels_low_sample.png?raw=true)

In terms of sentiment, the majority are positive. This is largely because the two labels - **value for money** and **garage service**

![alt text](https://github.com/sampathkethineedi/sentisum-topic-sentiment/blob/master/approach/pos_neg.png?raw=true)

I have under sampled these by removing a large fraction of single occurances. This also took care of the positive negative imbalance to an extent

Here is label distribution of the final dataset. We have **23 topic-sentiment labels**

![alt text](https://github.com/sampathkethineedi/sentisum-topic-sentiment/blob/master/approach/labels_all_2.png?raw=true)

We also observe that most exampels have less than 4 labels

![alt text](https://github.com/sampathkethineedi/sentisum-topic-sentiment/blob/master/approach/labels_len.png?raw=true)

| Number of Labels | Examples |
| --- | --- |
| 1 | 2007 |
| 2 | 3897 |
| 3 | 2705 |
| 4 | 1085 |
| 5 | 313 |
| 6 | 91 |
| 7 | 22 |
| 8 | 5 |
| 9 | 4 |
| 10 | 2 |

Most text examples are under 64 tokens with a few exceptions. This led to the tokenizer max length as 64

len mean 22.455 | len std 31.52 | len max 626

