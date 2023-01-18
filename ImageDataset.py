import torch
import pandas as pd
import transformers
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            # transforms.Resize(224),
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        label = torch.tensor(self.dataframe.iloc[idx, 3:11])
        img_path = "Cancer_Data/" + self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataframe)
