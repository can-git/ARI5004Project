import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import Properties as p
from Net import Net
from Preprocess import Preprocess
from tqdm import tqdm
from ImageDataset import ImageDataset


def train(model, train_data, val_data, criterion, optimizer):
    train, val = ImageDataset(train_data), ImageDataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=p.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=p.BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")



    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(p.EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_input, train_label = train_input.to(device), train_label.to(device)

            output = model(train_input)

            batch_loss = criterion(torch.log(output), train_label.argmax(dim=1))
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

                batch_loss = criterion(torch.log(output), val_label.argmax(dim=1))
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data):
    test = ImageDataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=p.BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            # print(output[0])

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    # new_df = pd.DataFrame({"img_id": names, "cancer_score": predictions})
    # new_df.to_csv("can_yilmaz_assignment_1.csv", index=False)

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


df_train, df_val, df_test = Preprocess().getItem()


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=p.LR)
train(model, df_train, df_val, criterion, optimizer)

# evaluate(model, df_test)
