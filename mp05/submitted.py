import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    # raise NotImplementedError("You need to write this part!")

    block = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 5)
    )

    return block



def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    # raise NotImplementedError("You need to write this part!")

    loss_function = torch.nn.CrossEntropyLoss()

    return loss_function


class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        # raise NotImplementedError("You need to write this part!")
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(2883, 140),
            torch.nn.ReLU(),
            torch.nn.Linear(140, 20),
        )
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        # raise NotImplementedError("You need to write this part!")
        y = self.linear_relu_stack(x)
        return y
        ################## Your Code Ends here ##################


def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer
    model = NeuralNet()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-3)
    loss_values = []

    # iterate over training set for epochs many times
    for epoch in range (epochs):
        # iterate and train over each batch 
        for features, labels in train_dataloader:
            # get the prediction and loss
            y_pred = model.forward(features)
            loss = loss_fn(y_pred, labels)

            # perform backpropogation and gradient descent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # track loss values 
            loss_values.append(loss)

    # raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################

    return model
