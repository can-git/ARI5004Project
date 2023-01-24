import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import seaborn as sns
import os
import numpy as np


class Utils_Plot:
    def __init__(self, plot_title):
        self.plot_title = plot_title
        if not os.path.exists("Visualization"):
            os.mkdir("Visualization")

    def plotConfusionMatrix(self, cm):
        sns.heatmap(cm, annot=True, fmt=".0f")
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(self.plot_title)
        plt.savefig("Results/{}/ConfusionMatrix.png".format(self.plot_title))
        plt.close()
        print("Results/{}/ConfusionMatrix saved".format(self.plot_title))

    def plotErrorBar(self, p):
        plt.barh(y=range(len(p)), width=p)
        # plt.yticks(range(len(p)), target_labels)
        plt.xlabel('Precision Score')
        plt.ylabel('Class')
        plt.show()

    def plotAuRoc(self, scores):
        for i, (score, fpr, tpr) in enumerate(scores):
            plt.plot(fpr, tpr, label=f'Class {i}, AUC = {score:0.5f}')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.plot_title)
        plt.savefig("Results/{}/AuRoc.png".format(self.plot_title))
        plt.close()
        print("Results/{}/AuRoc saved".format(self.plot_title))

    def plotCappa(self, score):
        plt.plot(['Cohen\'s Kappa'], [score], marker='o')
        plt.ylim([-1, 1])
        plt.ylabel('Score')
        plt.show()

    def plotSidesDistribution(self, data):
        sns.set_theme(style="ticks")

        f, ax = plt.subplots()
        sns.despine(f)

        sns.histplot(data, x="classes", hue="side", multiple="stack", edgecolor=".3", linewidth=.5, )
        plt.xlabel("Classes")
        plt.savefig("Results/SidesDistribution.png")
        plt.close()
        print("Results/SidesDistribution saved")

    def plotClassesDistribution(self, data):
        class_counts = data['classes'].value_counts()
        class_counts.plot(kind='bar')
        plt.ylabel('Count')
        plt.title(self.plot_title)
        plt.savefig("Results/ClassDistribution.png")
        plt.close()
        print("Results/ClassDistribution saved")

