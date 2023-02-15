import torch
import torch.nn as nn


class CNTK(nn.Module):
    def __init__(self, ntk_width=32):
        super(CNTK, self).__init__()
        self.conv1 = nn.Conv2d(3, ntk_width, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(ntk_width * 32 * 32, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = output.view(-1)
        # output = output.view(-1, ntk_width * 32 * 32)
        output = self.relu(output)
        output = self.fc(output)
        return output


class CNTK_2L(nn.Module):
    def __init__(self, ntk_width=32, ntk_width_2=32):
        super(CNTK_2L, self).__init__()
        self.conv1 = nn.Conv2d(3, ntk_width, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ntk_width, ntk_width_2, kernel_size=3, padding=1)
        self.bn = nn.LayerNorm(ntk_width_2 * 32 * 32)
        self.fc = nn.Linear(ntk_width_2 * 32 * 32, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        output = self.conv2(output)
        output = output.view(-1)
        output = self.bn(output)
        # output = output.view(-1, ntk_width_2 * 32 * 32)
        output = self.fc(output)
        return output


class CNTK_1D(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2, output_size=1):
        super(CNTK_1D, self).__init__()
        self.conv1d = nn.Conv1d(1, hidden_size1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(hidden_size1 * input_dim, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class CNTK_1D_1L(nn.Module):
    def __init__(self, input_dim, hidden_size1, hidden_size2, output_size=1):
        super(CNTK_1D_1L, self).__init__()
        self.conv1d = nn.Conv1d(1, hidden_size1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(hidden_size1 * input_dim, output_size)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)

        return x


class NTK_TL(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(NTK_TL, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
        self.layer_norm_1 = torch.nn.LayerNorm(self.hidden_size_1, elementwise_affine=False)
        self.layer_norm_2 = torch.nn.LayerNorm(self.hidden_size_2, elementwise_affine=False)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x, dummy=None):
        x = torch.flatten(x, 1)
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        # hidden = self.layer_norm_1(hidden)
        hidden = self.fc2(hidden)
        # output = self.relu(output)
        hidden = self.layer_norm_2(hidden)
        output = self.fc3(hidden)
        # output = self.softmax(output)

        return output


class NTK(nn.Module):

    def __init__(self, input_size, hidden_size_1, output_size):
        super(NTK, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = 0
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        # nn.init.xavier_uniform_(self.fc1.weight.data)
        # nn.init.uniform_(self.fc1.bias.data)
        self.relu = torch.nn.ReLU()
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size_1, elementwise_affine=False)
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.output_size)
        # nn.init.uniform_(self.fc2.weight.data)
        # nn.init.uniform_(self.fc2.bias.data)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x, dummy=None):
        x = torch.flatten(x, 1)
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        hidden = self.layer_norm(hidden)
        output = self.fc2(hidden)
        # output = self.softmax(output)

        return output
