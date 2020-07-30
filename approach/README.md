# Final Approach

I have approached this as a multi-label classification problem and used transformer based approach to fine tune a pre-trained BERT model. 

### Primary reasons for this
- Relatively small Dataset
- Niche Domain
- Leverage the information in language model
- Online Reviews are not rich in structure
- Ease of implementation

### Evaluation Metrics

FULL Dataset: 7757
TRAIN Dataset: 6205
VAL Dataset: 1552

Accuracy Score: 0.8200171821305842
Hamming Score: 0.9051907142113328
F1 Score (Micro) = 0.9182371701942591
F1 Score (Macro) = 0.7874733783966089
F1 Score (Weighted) = 0.9039791053690711

## Understanding the data

There are **62 unique topic-sentiment labels** (after fixing a few duplicates)

Most of the topics have both positive and negative samples

There are a lot of labels with less than a 100 samples. I have categorised these into two labels - **others positive** and **others negative**

In terms of sentiment, the majority are positive. This is largely because the two labels - **value for money** and **garage service**
I have under sampled these by removing a large fraction of single occurances. This also took care of the positive negative imbalance to an extent

We also observe that most exampels have less than 4 labels. So the Target is going to be sparse.

2007 example(s) with 1 labels
3897 example(s) with 2 labels
2705 example(s) with 3 labels
1085 example(s) with 4 labels
313 example(s) with 5 labels
91 example(s) with 6 labels
22 example(s) with 7 labels
5 example(s) with 8 labels
4 example(s) with 9 labels
2 example(s) with 10 labels

Coming to text

Text len mean 22.455586261350177
Text len std 31.524501540800433
Text len max 626
