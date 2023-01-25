import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ImageDataset import ImageDataset
import Utils_Metrics as um
import Utils_Plot as up
import csv
import os
import click
from Preprocess import Preprocess


class Evaluation:
    def __init__(self, model, test_data, model_name, im_size):
        self.im_size = im_size
        self.y_pred, self.y_prob, self.y = self.get_predictions(model, test_data)

        metrics = um.Utils_Metrics()
        metrics.set(self.y_pred, self.y_prob, self.y)
        plots = up.Utils_Plot(model_name)
        plots.plotAuRoc(metrics.get_auroc())
        plots.plotConfusionMatrix(metrics.get_confusion())

        # self.save_to_csv(model_name, metrics.get_acc(), metrics.get_precision(), metrics.get_f1(),
        #                      metrics.get_recall())

    def save_to_csv(self, mn, acc, pr, f1, rc):

        header = ['Model_Name', 'Accuracy', 'Precision', 'F1', 'Recall']

        # Open the CSV file for writing
        with open('Results/metrics.csv', 'a') as csvfile:
            # Create a CSV writer
            writer = csv.DictWriter(csvfile, fieldnames=header)

            # Write the header row if the file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write the data rows
            writer.writerow({'Model_Name': mn, 'Accuracy': acc, 'Precision': pr, 'F1': f1, 'Recall': rc})

        print("{} saved to file".format(mn))

    def data_visualization(self):
        if os.path.exists("Data/data.csv"):
            df = pd.read_csv("Data/data.csv")
            plots = up.Utils_Plot("Distribution of Train-Test-Validation")
            plots.plotSidesDistribution(df)
            plots.plotClassesDistribution(df)

    def get_predictions(self, model, test_data):
        test = ImageDataset(test_data, self.im_size)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
        model.eval()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        if use_cuda:
            model = nn.DataParallel(model, device_ids=[0, 3]).to(device)

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


@click.command()
@click.option('--name', '-n', default="densenet121", help='Name of the model.')
@click.option('--im_size', '-s', default=228, help='Image Size')
# @click.option('--visualization', '-v', is_flag=True, help="Export Data Visuals")
def main(name, im_size):
    preprocess = Preprocess()
    df_test = preprocess.getItem("test")
    Evaluation(torch.load("Results/{}/{}_model.pt".format(name, name)), df_test, name, im_size)


if __name__=="__main__":
    main()
