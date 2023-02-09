import torch
import torch.nn as nn


class NTK(nn.Module):

    def __init__(self, input_size, hidden_size_1, output_size):
        super(NTK, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = 0
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size_1, elementwise_affine=False)
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.output_size)

    def forward(self, x):
        x = torch.flatten(x, 0)
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        hidden = self.layer_norm(hidden)
        output = self.fc2(hidden)

        return output
