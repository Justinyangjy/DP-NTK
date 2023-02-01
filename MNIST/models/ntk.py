import torch
import torch.nn as nn
import torch.nn.functional as F


class CNTK(nn.Module):
    def __init__(self):
        super(CNTK, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = output.view(-1, 32 * 32 * 32)
        output = self.fc(output)
        return output


class Net_eNTK(nn.Module):
    def __init__(self):
        super(Net_eNTK, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(16, 128)
        self.layer_norm = torch.nn.LayerNorm(128, elementwise_affine=False)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.layer_norm(x)
        # x = F.relu(x)
        x = self.fc2(x)
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
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.relu(output)
        output = self.fc3(output)
        # output = self.softmax(output)

        return output


class NTK_TLNB(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(NTK_TLNB, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1, bias=False)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2, bias=False)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size, bias=False)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.relu(output)
        output = self.fc3(output)
        # output = self.softmax(output)

        return output


class NTK_NB(nn.Module):

    def __init__(self, input_size, hidden_size_1, output_size):
        super(NTK_NB, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = 0
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1, bias=False)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.output_size, bias=False)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        output = self.fc2(hidden)
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
        # nn.init.uniform_(self.fc1.weight.data)
        # nn.init.uniform_(self.fc1.bias.data)
        self.relu = torch.nn.ReLU()
        self.layer_norm = torch.nn.LayerNorm(self.hidden_size_1, elementwise_affine=False)
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.output_size)
        # nn.init.uniform_(self.fc2.weight.data)
        # nn.init.uniform_(self.fc2.bias.data)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = torch.flatten(x, 0)
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        hidden = self.layer_norm(hidden)
        output = self.fc2(hidden)
        # output = self.softmax(output)

        return output



class NTK_pretrain(nn.Module):

    def __init__(self, input_size, hidden_size_1, output_size):
        super(NTK_pretrain, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = 0
        self.output_size = output_size

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.output_size)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # print(x.flatten(1).shape)
        hidden = self.fc1(x.flatten(1))
        hidden = self.relu(hidden)
        hidden = self.fc2(hidden)
        output = nn.functional.log_softmax(hidden, dim=1)

        return output
