
'''
This is a new CNN that out performs all the previous cases
1. change the layers
2. adapt the neurons
3. adjust the learning rate to find the optimal weights
4. choose a good optimizer
5. choose a good loss function

# Goal: optimal performance

Plans:
1. choose binary cross entropy
2. choose adam for the optimizer
3. Fine-tune the layer first (2, 3)
4. Fine-tune the number of layers (6, 10, 20) etc.
5. check for the input convolutional layers

Strategy:
This is a binary classification problem and thus it is more simple (plane + dog)

1. high-level adaptation: Adam + BCE (idea) -- show the advantages
    a. two or three hidden layers
    b. less parameters for each layer
    c. fewer parameters than the previous ones
2. low-level systematic adaptation
    a. Find

'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from AI_Hw5 import NN_CNN
from torchsummary import summary

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()
        # Create the layers
        # self.conv = nn.Conv2d(3, 16, 9)
        # self.pool = nn.MaxPool2d(3)
        # self.fc = nn.Linear(1024, 1)

        # My solution -- 0.895 with optimizer = optim.Adam(cnn.parameters(), lr = alpha, weight_decay = 0.3)
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 10)
        # self.fc2 = nn.Linear(10, 1)

        # Update solution
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 5)
        self.fc1 = nn.Linear(8 * 5 * 5, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):

        # Forward pass
        # x = self.pool(torch.relu(self.conv(x)))
        # x = x.view(-1, 1024)
        # x = torch.sigmoid(self.fc(x))

        # My solution
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))

        # New solution
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

cnn = CNN()  # Construct the network

cuda = True
if cuda and torch.cuda.is_available():
    cnn = cnn.cuda()

weight_scale = 1e-6             # Maximum initial weight
alpha = 1e-3                    # Learning rate
T = 5000                        # Number of epochs
log = True                      # If False the prints will be disabled
dt = 100                        # Log the loss in every dt epochs
loss_train = np.empty(T // dt)  # Training loss
loss_test = np.empty(T // dt)   # Test loss
accuracy = np.empty(T // dt)    # Test accuracy

# Short variable names for brevity
X_train, X_test, y_train, y_test = NN_CNN.get_data()
x, x_ = X_train.reshape(-1, 3, 32, 32), X_test.reshape(-1, 3, 32, 32)

y, y_ = y_train, y_test

# The loss function and the optimization method
# Binary Cross Entropy
criterion = nn.BCELoss()

# The problem of SGD -- cases
# optimizer = optim.SGD(cnn.parameters(), lr = alpha, momentum = 0.9)

# The problem of Adam -- pending cases
# 6,893 > 4,929 (original parameters)
print("x.shape = ", x.shape)
print("check the number of paraemeters ", summary(cnn, input_size = (3, 32, 32)))
input("check")

optimizer = optim.Adam(cnn.parameters(), lr = alpha, weight_decay = 0.3)

# Start training
start = time.time()
for t in range(T):

    optimizer.zero_grad()

    # forward + backward
    z = cnn(x)[:, 0]
    loss = criterion(z, y)
    loss.backward()

    # Print the training loss, the test loss and the test accuracy
    if (t + 1) % dt == 0:

        # Store the training loss
        loss_train[t // dt] = loss

        # Predict the classes of the images in the test set
        z = cnn(x_)[:, 0]
        loss_test[t // dt] = F.binary_cross_entropy(z, y_)
        accuracy[t // dt] = 1 - torch.mean(torch.abs((z > 0.5).float() - y_))

        if log:
            print('Epoch:', t + 1)
            print('Training Loss:', loss_train[t // dt])
            print('Test Loss:', loss_test[t // dt])
            print('Testing Accuracy:', accuracy[t // dt])

    # Update the weights
    optimizer.step()

end = time.time()

# Print the statistics
print('Training time:', end - start)
print('Minimum Training Loss:', np.min(loss_train))
print('Minimum Test Loss:', np.min(loss_test))
print('Maximum Accuracy:', np.max(accuracy))