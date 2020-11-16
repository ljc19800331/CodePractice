
'''
Test the Two layer network

'''

import torch.nn.functional
from Code_DL import NN_TwoLayer

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 2500, 3072, 6, 1
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# The homework data
n_hidden = 6
weight_scale = 1e-4 * 1.0
x, x_, y, y_ = NN_TwoLayer.get_data()
N, D_in, H, D_out = x.shape[0], x.shape[1], n_hidden, 1

# Model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    # torch.nn.ReLU(),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, D_out),
    # torch.nn.Sigmoid(),
)

model.to("cuda:0")

# Define the loss function
loss_fn = torch.nn.MSELoss(reduction = 'sum')
# loss_fn = torch.nn.CrossEntropyLoss()

# The learning rate
learning_rate = 7e-2

for t in range(5000):

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad