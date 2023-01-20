import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import Properties as p
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.Resize((p.IMAGE_SIZE, p.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        ls = []
        for rows in self.dataframe.iloc[:, 3:11].values:
            for i, cols in enumerate(rows):
                if cols == 1:
                    ls.append(i)
        self.dataframe["labels"] = ls

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx, -1]
        img_path = "Cancer_Data/" + self.dataframe.iloc[idx, 0]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataframe)
