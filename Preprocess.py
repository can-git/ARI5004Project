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


class Preprocess:
    def __init__(self):

        if not os.path.exists("Cancer_Data"):
            os.mkdir("Cancer_Data")

            raw_df = pd.read_csv("Folds.csv")
            raw_df = raw_df.loc[raw_df['mag'] == 40]
            raw_df = raw_df.iloc[:, -1:]
            self.move_images(raw_df)

            raw_df['filename'] = raw_df['filename'].apply(lambda x: x.split("/")[-1])

            malignant_or_not = []
            classes = []
            for file_name in raw_df["filename"]:
                image_details = file_name.split("-")
                details = image_details[0].split("_")
                malignant_or_not.append(details[1])
                classes.append(details[2])
            df = pd.DataFrame({
                'filename': raw_df["filename"],
                'malignant': malignant_or_not,
                'classes': classes})
            df = df.reset_index(drop=True)

            df = pd.get_dummies(df, columns=['malignant'])
            df = pd.get_dummies(df, columns=['classes'])

            df_train, df_val_test = train_test_split(df, test_size=0.3)
            df_test, df_val = train_test_split(df_val_test, test_size=0.5)

            df_train["side"] = ['train' for i in range(len(df_train))]  # 6982
            df_test["side"] = ['test' for i in range(len(df_test))]  # 1496
            df_val["side"] = ['val' for i in range(len(df_val))]  # 1497
            new_df = pd.concat([df_train, df_test, df_val])
            new_df.to_csv("Data.csv", index=False)

    def getItem(self):
        global df_train, df_val, df_test
        if os.path.exists("Data.csv"):
            df = pd.read_csv("Data.csv")
            df_train = df.loc[df['side'] == "train"]
            df_test = df.loc[df['side'] == "test"]
            df_val = df.loc[df['side'] == "val"]
        return df_train, df_val, df_test

    def move_images(self, df):
        for row in df["filename"]:
            shutil.copy(row, "Cancer_Data/")
