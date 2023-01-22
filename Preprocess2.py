import re
import string
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Properties as p
import random
import shutil
import os
import Utils_Plot as up


class Preprocess2:
    def __init__(self):

        if not os.path.exists("ValData"):
            os.mkdir("ValData")
            file_names = []
            labels = []
            for folder in os.listdir("val"):
                for image in os.listdir("val/{}".format(folder)):
                    file_names.append(image.split(".")[0])
                    labels.append(folder)

            df = pd.DataFrame({'file_names': file_names, 'classes': labels})

            classes = df["classes"]
            df = pd.get_dummies(df, columns=['classes'])
            df["classes"] = classes

            for i in df["classes"].unique():
                df_classes = df[df['classes'] == i]
                df = df.drop(df[df['classes'] == i].index, inplace=False)
                df_classes = df_classes.iloc[:4000]
                df = pd.concat([df, df_classes])

            self.move_images(df)

            df_train, df_val_test = train_test_split(df, test_size=0.3, random_state=1)
            df_test, df_val = train_test_split(df_val_test, test_size=0.5, random_state=1)

            df_train["side"] = ['train' for i in range(len(df_train))]  # 14000
            df_test["side"] = ['test' for i in range(len(df_test))]  # 3000
            df_val["side"] = ['val' for i in range(len(df_val))]  # 3000
            new_df = pd.concat([df_train, df_test, df_val])
            new_df.to_csv("val.csv", index=False)


    def getItem(self):
        global df_train, df_val, df_test
        if os.path.exists("val.csv"):
            df = pd.read_csv("val.csv")
            df_train = df.loc[df['side'] == "train"]
            df_test = df.loc[df['side'] == "test"]
            df_val = df.loc[df['side'] == "val"]
        return df_train, df_val, df_test

    def show(self):
        if os.path.exists("val.csv"):
            df = pd.read_csv("val.csv")
            plots = up.Utils_Plot("Distribution of Train-Test-Validation")
            plots.plotSidesDistribution(df)
            plots.plotClassesDistribution(df)
            # plots.plotMalignantDistribution(df)

    def move_images(self, df):
        for i, row in enumerate(df.values):
            shutil.copy("val/{}/{}.jpg".format(row[-1], row[0]), "ValData")

