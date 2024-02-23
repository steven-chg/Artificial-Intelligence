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
        
        # FOLLOWING CONVOLUTION CODE REFERENCES CAMPUSWIRE POSTS
        # self.conv2d = torch.nn.Sequential(
        #     # torch.nn.Conv2d(3, 16, 5),
        #     # torch.nn.Conv2d(16, 32, 5),             
        #     # torch.nn.AdaptiveAvgPool2d((6,6))
        #     torch.nn.functional.max_pool2d(torch.nn.functional.relu(torch.nn.Conv2d(3, 16, 5)), (6, 6)),     
        #     torch.nn.functional.max_pool2d(torch.nn.functional.relu(torch.nn.Conv2d(16, 32, 5)), (6, 6)) 
        # )

        self.conv1 = torch.nn.Conv2d(3, 32, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 2)

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(32*5*5, 160),
            torch.nn.ReLU(),
            torch.nn.Linear(160, 5)
        )

        self.linear_relu_last = torch.nn.Sequential(
            torch.nn.Linear(2883, 160),
            torch.nn.ReLU(),
            torch.nn.Linear(160, 5)        
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

        if x.shape != torch.Size([100, 2883]):
            y = self.linear_relu_last(x)
        
        else:
            # FOLLOWING CODE REFERENCES CAMPUSWIRE POSTS
            # unflatten for 2d convolution; 100 - batch size; 3 - number of input channels; 31, 31 - each channel has size 31x31
            x = x.reshape(100, 3, 31, 31)

            # perform convolution
            x = self.conv1(x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.max_pool2d(x, (6, 6))
            # print(x.shape)
            # x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), (6, 6))

            # # flatten all dimensions except batch dimension
            x = torch.flatten(x, 1)

            # apply linear relu linear 
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
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    loss_values = []

    # iterate over training set for epochs many times
    for epoch in range (epochs):
        # iterate and train over each batch 
        for features, labels in train_dataloader:
            # get the prediction and loss
            y_pred = model.forward(features)
            loss = loss_fn(y_pred, labels)

            # perform backpropogation and gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track loss values 
            loss_values.append(loss)

    # raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################

    return model
