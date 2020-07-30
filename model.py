import torch
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, AutoTokenizer, BertConfig
import numpy as np
from sklearn import metrics
from typing import Optional
import pickle
import os
import config

config = config.Settings()


class BERTClass(BertPreTrainedModel):
    def __init__(self, model_config):
        super(BERTClass, self).__init__(model_config)
        self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, config.NUM_LABELS)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class Trainer:
    def __init__(self, path: str, load: bool = False):
        self.device = config.DEVICE
        self.global_epochs = 0
        self.optimizer = None
        self.loss_fun = None

        if load:
            self.load(path)
        else:
            self.model = None
            self.tokenizer = None

    def prepare(self, model: Optional, optimizer, loss_fun):
        if not self.model:
            self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun

    def train(self, epochs, training_loader, testing_loader=None, validate=False):
        for epoch in range(epochs):
            self.model.train()
            for _, data in enumerate(training_loader, 0):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids, mask, token_type_ids)

                self.optimizer.zero_grad()
                loss = self.loss_fun(outputs, targets)
                if _ % 200 == 0:
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.global_epochs += 1
            if validate:
                self.validate(testing_loader)

    def validate(self, testing_loader):
        self.model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['input_ids'].to(self.device, dtype=torch.long)
                mask = data['attention_mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['labels'].to(self.device, dtype=torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        fin_outputs = np.array(fin_outputs) >= 0.5
        accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
        f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro')
        f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        print("------------------------------------")

    def save_model(self, path: str):
        torch.save(self.model, path)

    def save_all(self, path: str, tokenizer: BertTokenizer, label_encoder):
        torch.save(self.model, path)
        tokenizer.save_pretrained(path)
        output = open(os.path.join(path, 'label_encoder.pkl', 'wb'))
        pickle.dump(label_encoder, output)
        output.close()

    def predict(self, text: str, tokenizer: Optional, label_encoder, threshold: float = 0.2, max_len: int = 64):

        if not self.tokenizer:
            self.tokenizer = tokenizer

        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=max_len,
            add_special_tokens=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )

        encoded_text.to(self.device)

        output = self.model(encoded_text["input_ids"], encoded_text["attention_mask"],
                            encoded_text["token_type_ids"]).sigmoid()

        prediction = [1 if i > threshold else 0 for i in output[0]]

        label = label_encoder.inverse_transform(np.array([prediction]))[0]

        print(f'Review text: {text}')
        print(f'Prediction  : {label}')

        return label

    def load(self, path: str):
        model_config = BertConfig()
        self.model = BERTClass(model_config)
        self.model.load_state_dict(torch.load(path, map_location=config.DEVICE))

        tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL)

        return "Model load from '{}' complete".format(path)


class FocalLossLogits(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLossLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.BCEWithLogitsLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        # noinspection PyTypeChecker
        f_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return torch.mean(f_loss)
