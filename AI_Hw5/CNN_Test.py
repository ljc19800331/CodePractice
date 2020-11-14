import torch
import torch.nn as nn
from torchsummary import summary

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()
        # Create the layers
        self.conv = nn.Conv2d(3, 16, 9)
        self.pool = nn.MaxPool2d(3)
        self.fc = nn.Linear(1024, 1)

        # My solution -- 0.895 with optimizer = optim.Adam(cnn.parameters(), lr = alpha, weight_decay = 0.3)
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 10)
        # self.fc2 = nn.Linear(10, 1)

        # Update solution
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 8, 5)
        # self.fc1 = nn.Linear(8 * 5 * 5, 10)
        # self.fc2 = nn.Linear(10, 1)

    def forward(self, x):

        # Forward pass
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 1024)
        x = torch.sigmoid(self.fc(x))

        # My solution
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))

        # New solution
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 8 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))

        return x

cnn = CNN()  # Construct the network
print("check the number of paraemeters ", summary(cnn, input_size = (3, 32, 32)))