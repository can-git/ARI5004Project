import re
import string
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import Properties as p
import random
import shutil


class Preprocess:
    def __init__(self):
        raw_df = pd.read_csv(p.DEFAULT_PATH_CSV)

        raw_df_40 = raw_df.query("mag == 40")
        # self.move_images(raw_df_40)
        raw_df_40['filename'] = raw_df_40['filename'].apply(lambda x: x.split("/")[-1])
        file_names = raw_df_40["filename"]
        malignant_or_not = []
        classes = []
        for file_name in file_names:
            image_details = file_name.split("-")
            details = image_details[0].split("_")
            malignant_or_not.append(details[1])
            classes.append(details[2])
        df = pd.DataFrame({
            'filename': file_names,
            'malignant': malignant_or_not,
            'classes': classes})
        df = df.reset_index(drop=True)

        df = pd.get_dummies(df, columns=['malignant'])
        df = pd.get_dummies(df, columns=['classes'])

        self.df_train, df_val_test = train_test_split(df, test_size=0.3, random_state=1)
        self.df_val, self.df_test = train_test_split(df_val_test, test_size=0.5, random_state=1)

    def getItem(self):
        return self.df_train, self.df_val, self.df_test

    def move_images(self, df):
        for row in df["filename"]:
            shutil.copy(row, "Cancer_Data/")
