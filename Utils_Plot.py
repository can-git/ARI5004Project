import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sn
import os


class Utils_Plot:
    def __init__(self, plot_title):
        self.plot_title = plot_title
        if not os.path.exists("Visualization"):
            os.mkdir("Visualization")

    def plotConfusionMatrix(self, cm):
        sn.heatmap(cm, annot=True, fmt=".0f")
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
        plt.show()

    def plotCappa(self, score):
        plt.plot(['Cohen\'s Kappa'], [score], marker='o')
        plt.ylim([-1, 1])
        plt.ylabel('Score')
        plt.show()

    def plotClassesDistribution(self, data):

        class_counts = data['side'].value_counts()
        class_counts.plot(kind='bar')
        plt.ylabel('Count')
        plt.title(self.plot_title)
        plt.savefig("Visualization/distribution.png")
        plt.close()
