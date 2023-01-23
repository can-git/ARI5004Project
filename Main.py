import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torchvision import models
import Properties as p
from Net import Net
import os
from Preprocess import Preprocess
from tqdm import tqdm
from ImageDataset import ImageDataset
from Evaluation import Evaluation
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class Main:
    def __init__(self, version):
        self.version = version

        preprocess = Preprocess()
        # preprocess.show()
        df_train, df_val, df_test = preprocess.getItem()

        if self.version == "cnn8":
            model = Net()
        elif self.version == "resnet18":
            model = getattr(models, version)()
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, p.NUM_CLASSES)
        elif self.version == "densenet121":
            model = models.densenet121()
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, p.NUM_CLASSES)
            torch.nn.init.kaiming_normal_(model.classifier.weight)


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=p.LR, weight_decay=p.WD)
        if p.TRAIN:
            self.train(model, df_train, df_val, criterion, optimizer)
        else:
            model.load_state_dict(torch.load("Results/{}/{}_model.pt".format(self.version, self.version)))
            Evaluation(model, df_test, self.version)

    def train(self, model, train_data, val_data, criterion, optimizer):
        train, val = ImageDataset(train_data), ImageDataset(val_data)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=p.BATCH_SIZE, shuffle=True,
                                                       num_workers=p.NUM_WORKERS, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=p.BATCH_SIZE, num_workers=p.NUM_WORKERS,
                                                     pin_memory=True)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        # device = torch.device("cpu")

        if use_cuda:

            model = nn.DataParallel(model, device_ids=[0]).cuda()

            criterion = criterion.cuda()

        for epoch_num in range(p.EPOCHS):
            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_input, train_label = train_input.to(device), train_label.to(device)

                output = model(train_input)

                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():
                for val_input, val_label in val_dataloader:
                    val_input, val_label = val_input.to(device), val_label.to(device)

                    output = model(val_input)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            print(
                f'\nEpochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_dataloader) * p.BATCH_SIZE): .3f} \
                    | Train Accuracy: {total_acc_train / (len(train_dataloader) * p.BATCH_SIZE): .3f} \
                    | Val Loss: {total_loss_val / (len(val_dataloader) * p.BATCH_SIZE): .3f} \
                    | Val Accuracy: {total_acc_val / (len(val_dataloader) * p.BATCH_SIZE): .3f}')

        if p.SAVE_MODEL:
            if not os.path.exists("Results"):
                os.mkdir("Results")
                if not os.path.exists("Results/{}".format(self.version)):
                    os.mkdir("Results/{}".format(self.version))
                    torch.save(model.state_dict(), "Results/{}/{}_model.pt".format( self.version, self.version))


if __name__ == "__main__":
    dl_models = ['resnet18', 'cnn8', 'densenet121']
    for version in dl_models:
        app = Main(version)
