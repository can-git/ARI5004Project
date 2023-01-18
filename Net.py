import torch
import torch.nn as nn
import torch.nn.functional as F



# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define 2D convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=634752, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=8)
        # Define dropout layer
        self.dropout = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass input through convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 634752)
        # Pass input through fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return self.softmax(x)