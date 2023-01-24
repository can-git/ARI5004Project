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
import click
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class Main:
    def __init__(self, name, batch_size, num_workers, epochs, lr, wd, gamma, save_model, im_size):
        self.version = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.gamma = gamma
        self.save_model = save_model
        self.im_size = im_size

        preprocess = Preprocess()
        df_train, df_val = preprocess.getItem("train")

        if self.version == "cnn8":
            model = Net()
        elif self.version == "resnet18":
            model = getattr(models, self.version)()
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, p.NUM_CLASSES)
        elif self.version == "densenet121":
            model = models.densenet121()
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, p.NUM_CLASSES)
            torch.nn.init.kaiming_normal_(model.classifier.weight)
        else:
            print("This model is not supported right now, see options with --help command")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.train(model, df_train, df_val, criterion, optimizer)

    def train(self, model, train_data, val_data, criterion, optimizer):
        train, val = ImageDataset(train_data), ImageDataset(val_data)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=self.num_workers, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=self.batch_size, num_workers=self.num_workers,
                                                     pin_memory=True)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        # device = torch.device("cpu")

        if use_cuda:
            model = model.cuda()

            criterion = criterion.cuda()

        for epoch_num in range(self.epochs):
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
                f'\nEpochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_dataloader) * self.batch_size): .3f} \
                    | Train Accuracy: {total_acc_train / (len(train_dataloader) * self.batch_size): .3f} \
                    | Val Loss: {total_loss_val / (len(val_dataloader) * self.batch_size): .3f} \
                    | Val Accuracy: {total_acc_val / (len(val_dataloader) * self.batch_size): .3f}')

        if p.SAVE_MODEL:
            if not os.path.exists("Results"):
                os.mkdir("Results")
            if not os.path.exists("Results/{}".format(self.version)):
                os.mkdir("Results/{}".format(self.version))
            torch.jit.script(model).save("Results/{}/{}_model.pt".format(self.version, self.version))


@click.command()
@click.option('--name', default="resnet18", help='Name of the model. (cnn8, resnet18 or densenet121)')
@click.option('--batch_size', default=1, help='Batch Size')
@click.option('--num_workers', default=12, help='Num Workers')
@click.option('--epochs', default=1, help='Epochs')
@click.option('--lr', default=0.00008, help='Learning Rate')
@click.option('--wd', default=0, help='Weight Decay')
@click.option('--gamma', default=0.9, help='Gamma')
@click.option('--save_model', default=True, help='Save Model at the end')
@click.option('--im_size', default=228, help='Image Size')
def main(name, batch_size, num_workers, epochs, lr, wd, gamma, save_model, im_size):
    Main(name, batch_size, num_workers, epochs, lr, wd, gamma, save_model, im_size)


if __name__ == "__main__":
    main()

