import torch
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import os
import config

config = config.Settings()


class BertForMultiLabel(BertPreTrainedModel):
    def __init__(self, model_config):
        super(BertForMultiLabel, self).__init__(model_config)
        self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL)
        self.drp = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, config.NUM_LABELS)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.drp(output_1)
        output = self.classifier(output_2)
        return output


class Trainer:
    def __init__(self, model, optimizer, loss_fun):
        self.device = config.DEVICE
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.metrics = {
            "epoch_loss": [],
            "step_loss": [],
            "f1_score_macro": [],
            "f1_score_micro": [],
            "accuracy": []
        }
        self.global_epochs = 0

    def train(self, epochs, training_loader, testing_loader=None, validate=False):
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for _, data in enumerate(training_loader, 0):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(ids, mask, token_type_ids)

                self.optimizer.zero_grad()
                loss = self.loss_fun(outputs, targets)
                epoch_loss = loss.item()
                if _ % 200 == 0:
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                    self.metrics["step_loss"].append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.global_epochs += 1
            if validate:
                self.validate(testing_loader)

            self.metrics["loss"].append(epoch_loss)

        return self.metrics

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
        self.metrics["accuracy"].append(accuracy)
        self.metrics["f1_score_micro"].append(f1_score_micro)
        self.metrics["f1_score_macro"].append(f1_score_macro)
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        print("------------------------------------")
        return self.metrics

    def save_model(self, path: str):
        torch.save(self.model, path)

    def save_all(self, path: str, tokenizer: BertTokenizer, label_encoder):
        torch.save(self.model, path)
        tokenizer.save_pretrained(path)
        output = open(os.path.join(path, 'label_encoder.pkl', 'wb'))
        pickle.dump(label_encoder, output)
        output.close()

    def plot_metrics(self, metric: str):
        plt.plot(range(len(self.metrics[metric])), self.metrics[metric], '-b', label=metric)

        plt.xlabel("Epochs")
        plt.legend(loc='upper left')
        plt.title(metric)
        plt.savefig(os.path.join(config.MODEL_DIR, metric + ".png"))
        plt.show()


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
