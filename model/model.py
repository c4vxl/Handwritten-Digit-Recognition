import torch.nn as nn
import torch.nn.functional as F

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, n_inp, n_out = 10, n_layer = 3, layer_size = 8) -> None:
        super().__init__()

        # fully connected layers
        self.inp_layer = nn.Linear(n_inp, layer_size)
        self.hidden_layer = nn.ModuleList([nn.Linear(layer_size, layer_size) for _ in range(n_layer)])
        self.out_layer = nn.Linear(layer_size, n_out)

        self.act_f = nn.ReLU() # using relu as activation function
    
    def forward(self, x):
        """
        Pass tensor x through the nn
        """

        x.view(x.size(0), -1) # Flatten the input if it is not already flattened

        # pass through layers & apply activation function
        x = self.act_f(self.inp_layer(x)) # input layers
        for layer in self.hidden_layer: x = self.act_f(layer(x)) # hidden layers
        x = self.act_f(self.out_layer(x)) # output layers

        return x