import os
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from itertools import combinations
import pandas as pd


class Utils_Metrics:
    def set(self, y_pred, y_prob, y):
        self.y = y
        self.y_pred = y_pred
        self.y_prob = y_prob

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

        auc_roc_scores = []

        for i in range(y_true.shape[1]):
            y_true_binary = y_true[:, i]
            y_pred_binary = y_pred[:, i]
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
            roc_auc = auc(fpr, tpr)
            # auc_roc = roc_auc_score(y_true_binary, y_pred_binary)
            auc_roc_scores.append((roc_auc, fpr, tpr))

        return auc_roc_scores

    def get_cappa(self, model_names, predictions, true_labels):
        # scores = []
        # models = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]

                # scores.append(cohen_kappa_score(predictions[i], predictions[j], labels=true_labels))
                # models.append("{}_{}".format(model1, model2))
                print("Models ({}_{})'s kappa score is {}".format(model1, model2,
                                                                  cohen_kappa_score(predictions[i], predictions[j],
                                                                                    labels=true_labels)))

        # return pd.DataFrame({'model_names': models, 'kappa_scores': scores})
