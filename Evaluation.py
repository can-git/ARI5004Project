import re
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Properties as p
import random
import shutil
import torch
import torch.nn as nn
from ImageDataset import ImageDataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer


class Evaluation:
    def __init__(self, model, test_data, criterion):
        self.y_pred, self.y_prob, self.y = self.get_predictions(model, test_data)
        acc = self.get_acc()
        cm = self.get_confusion()
        pre = self.get_precision()
        rec = self.get_recall()
        f1 = self.get_f1()
        auroc = self.get_auroc()
        print(acc)
        print(cm)
        print(pre)
        print(rec)
        print(f1)
        print(auroc)

    def get_acc(self):
        return accuracy_score(self.y, self.y_pred)

    def get_confusion(self):
        return confusion_matrix(self.y, self.y_pred)

    def get_precision(self):
        return precision_score(self.y, self.y_pred, average='macro')

    def get_recall(self):
        return recall_score(self.y, self.y_pred, average='macro')

    def get_f1(self):
        return f1_score(self.y, self.y_pred, average='macro')

    def get_auroc(self):
        y = np.concatenate(self.y)
        y_pred = np.concatenate(self.y_prob, axis=0)

        lb = LabelBinarizer()
        y_true = lb.fit_transform(y)

        # Initialize the AUC-ROC score list
        auc_roc_scores = []

        # Loop through the classes
        for i in range(y_true.shape[1]):
            y_true_binary = y_true[:, i]
            y_pred_binary = y_pred[:, i]
            auc_roc = roc_auc_score(y_true_binary, y_pred_binary)
            auc_roc_scores.append(auc_roc)

        return auc_roc_scores

    def get_predictions(self, model, test_data):
        test = ImageDataset(test_data)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            model = model.cuda()

        predicted_values = []
        probabilities = []
        real_values = []
        with torch.no_grad():
            for test_input, test_label in test_dataloader:
                test_input, test_label = test_input.to(device), test_label.to(device)

                predicted = model(test_input)
                pred = predicted.detach().cpu().numpy()
                probability = np.exp(pred) / np.sum(np.exp(pred))
                predicted_values.append(predicted.argmax(dim=1).detach().cpu().numpy())
                probabilities.append(probability)
                real_values.append(test_label.detach().cpu().numpy())

        return predicted_values, probabilities, real_values
