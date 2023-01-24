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


class Visualization:
    def __init__(self, test_data, im_size, kappa):
        if not os.path.exists("Visualization"):
            os.mkdir("Visualization")

        self.im_size = im_size
        plots = up.Utils_Plot("Distribution of Train-Test-Validation")
        if os.path.exists("Data/data.csv"):
            df = pd.read_csv("Data/data.csv")
            plots.plotSidesDistribution(df)
            plots.plotClassesDistribution(df)
        metrics = um.Utils_Metrics()

        if kappa:
            preds = []
            label = []
            names = []
            for model in os.listdir("Results/"):
                if model != ".DS_Store":
                    if os.path.isdir(os.path.join("Results", model)):
                        pred, _, y = self.get_predictions(torch.jit.load("Results/{}/{}_model.pt".format(model, model)),
                                                          test_data)
                        preds.append(pred)
                        label = y
                        names.append(model)
            plots.plotCappa(metrics.get_cappa(names, preds, label))

    def get_predictions(self, model, test_data):
        test = ImageDataset(test_data, self.im_size)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

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
                predicted_values.append(predicted.argmax(dim=1).detach().cpu().numpy().tolist())
                probabilities.append(probability)
                real_values.append(test_label.detach().cpu().numpy())

        return np.ravel(predicted_values), probabilities, np.ravel(real_values)


@click.command()
@click.option('--im_size', '-s', default=228, help='Image Size')
@click.option('--kappa', '-k', is_flag=True, help="Export Kappa")
def main(im_size, kappa):
    preprocess = Preprocess()
    df_test = preprocess.getItem("test")
    Visualization(df_test, im_size, kappa)


if __name__ == "__main__":
    main()
