import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import Properties as p
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, dataframe, im_size):
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize((im_size, im_size)),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            # transforms.RandomRotation(degrees=90),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        ls = []
        for rows in self.dataframe.iloc[:, 1:6].values:
        # for rows in self.dataframe.iloc[:, 3:11].values:
            for i, cols in enumerate(rows):
                if cols == 1:
                    ls.append(i)
        self.dataframe["labels"] = ls

    def __getitem__(self, idx):
        label = self.dataframe.iloc[idx, -1]
        img_path = "Data/Lung_Data/" + self.dataframe.iloc[idx, 0]
        image = Image.open(img_path + ".jpg")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataframe)
