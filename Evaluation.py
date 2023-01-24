import re
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Properties as p
import random
import matplotlib.pyplot as plt
import shutil
import torch
import torch.nn as nn
from ImageDataset import ImageDataset
import Utils_Metrics as um
import Utils_Plot as up
import csv
import os


class Evaluation:
    def __init__(self, model, test_data, model_name):

        self.y_pred, self.y_prob, self.y = self.get_predictions(model, test_data)

        metrics = um.Utils_Metrics(self.y_pred, self.y_prob, self.y)
        plots = up.Utils_Plot(model_name)

        self.save_to_csv(model_name, metrics.get_acc(), metrics.get_precision(), metrics.get_f1(),
                             metrics.get_recall(), metrics.get_cappa())

        plots.plotConfusionMatrix(metrics.get_confusion())
        plots.plotAuRoc(metrics.get_auroc())
        self.data_visualization()

    def save_to_csv(self, mn, acc, pr, f1, rc, kap):

        header = ['Model_Name', 'Accuracy', 'Precision', 'F1', 'Recall', 'Kappa']

        # Open the CSV file for writing
        with open('Results/metrics.csv', 'a') as csvfile:
            # Create a CSV writer
            writer = csv.DictWriter(csvfile, fieldnames=header)

            # Write the header row if the file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write the data rows
            writer.writerow({'Model_Name': mn, 'Accuracy': acc, 'Precision': pr, 'F1': f1, 'Recall': rc, 'Kappa': kap})

        print("{} saved to file".format(mn))

    def data_visualization(self):
        if os.path.exists("Data/data.csv"):
            df = pd.read_csv("Data/data.csv")
            plots = up.Utils_Plot("Distribution of Train-Test-Validation")
            plots.plotSidesDistribution(df)
            plots.plotClassesDistribution(df)

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
