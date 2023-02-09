import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = input[0]
    return hook

class CNTK(nn.Module):
    def __init__(self, ntk_width=32):
        super(CNTK, self).__init__()
        self.conv1 = nn.Conv2d(3, ntk_width, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(ntk_width * 32 * 32, 10)

    def forward(self, x):
        output = self.conv1(x)
        output = output.view(-1)
        #output = output.view(-1, ntk_width * 32 * 32)
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
        output =self.relu(output)
        output = self.conv2(output)
        output = output.view(-1)
        output = self.bn(output)
        #output = output.view(-1, ntk_width_2 * 32 * 32)
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


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x): return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x): return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                        stride=stride, padding=padding, groups=groups, bias=False),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(inplace=True)
    )


def get_ffcv_model(device, num_class=1000):
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=torch.channels_last).to(device)
    return model


class Net_eNTK_pretrain(nn.Module):
    def __init__(self):
        super(Net_eNTK_pretrain, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(4096, 10)
        # self.layer_norm = torch.nn.LayerNorm(128, elementwise_affine=False)
        # self.fc2 = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        # x = F.relu(x)
        # x = self.conv4(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.layer_norm(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

class Net_eNTK(nn.Module):
    def __init__(self):
        super(Net_eNTK, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 4, kernel_size=3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(1024, 8192)
        # self.layer_norm = torch.nn.LayerNorm(128, elementwise_affine=False)
        self.fc2 = nn.Linear(8192, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.layer_norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = nn.functional.log_softmax(x, dim=1)
        return x

# class Net_eNTK(nn.Module):
#     def __init__(self):
#         super(Net_eNTK, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
#         # self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
#         # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
#         # self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
#         # self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
#         self.fc1 = nn.Linear(256, 1000)
#         # self.layer_norm = torch.nn.LayerNorm(128, elementwise_affine=False)
#         self.fc2 = nn.Linear(1000, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         # x = F.relu(x)
#         # x = self.conv2(x)
#         # x = F.relu(x)
#         # x = self.conv3(x)
#         # x = F.relu(x)
#         # x = self.conv4(x)
#         # x = F.relu(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         # x = self.layer_norm(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x


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
