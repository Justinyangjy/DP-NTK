from collections import OrderedDict
from math import log

import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
from util_logging import LOG


class FCCondGen(nn.Module):
    def __init__(self, d_code, d_hid, d_out, n_labels, use_sigmoid=True, batch_norm=True):
        super(FCCondGen, self).__init__()
        d_hid = [int(k) for k in d_hid.split(',')]
        assert len(d_hid) < 5

        self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
        self.fc2 = nn.Linear(d_hid[0], d_hid[1])

        self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(d_hid[1]) if batch_norm else None
        if len(d_hid) == 2:
            self.fc3 = nn.Linear(d_hid[1], d_out)
        elif len(d_hid) == 3:
            self.fc3 = nn.Linear(d_hid[1], d_hid[2])
            self.fc4 = nn.Linear(d_hid[2], d_out)
            self.bn3 = nn.BatchNorm1d(d_hid[2]) if batch_norm else None
        elif len(d_hid) == 4:
            self.fc3 = nn.Linear(d_hid[1], d_hid[2])
            self.fc4 = nn.Linear(d_hid[2], d_hid[3])
            self.fc5 = nn.Linear(d_hid[3], d_out)
            self.bn3 = nn.BatchNorm1d(d_hid[2]) if batch_norm else None
            self.bn4 = nn.BatchNorm1d(d_hid[3]) if batch_norm else None

        self.use_bn = batch_norm
        self.n_layers = len(d_hid)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self.d_code = d_code
        self.n_labels = n_labels

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x) if self.use_bn else x
        x = self.fc2(self.relu(x))
        x = self.bn2(x) if self.use_bn else x
        x = self.fc3(self.relu(x))
        if self.n_layers > 2:
            x = self.bn3(x) if self.use_bn else x
            x = self.fc4(self.relu(x))
            if self.n_layers > 3:
                x = self.bn4(x) if self.use_bn else x
                x = self.fc5(self.relu(x))

        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x

    def get_code(self, batch_size, device, return_labels=True, labels=None):
        if labels is None:  # sample labels
            labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
        code = pt.randn(batch_size, self.d_code, device=device)
        gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
        gen_one_hots.scatter_(1, labels, 1)
        code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
        # print(code.shape)
        if return_labels:
            return code, gen_one_hots
        else:
            return code


class ConvCondGen(nn.Module):
    def __init__(self, d_code, d_hid, n_labels, nc_str, ks_str, use_sigmoid=True, batch_norm=True):
        super(ConvCondGen, self).__init__()
        self.nc = [int(k) for k in nc_str.split(',')] + [1]  # number of channels
        self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
        d_hid = [int(k) for k in d_hid.split(',')]
        assert len(self.nc) == 3 and len(self.ks) == 2
        self.hw = 7  # image height and width before upsampling
        self.reshape_size = self.nc[0] * self.hw ** 2
        self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
        self.fc2 = nn.Linear(d_hid[0], self.reshape_size)
        self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(self.reshape_size) if batch_norm else None
        self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0] - 1) // 2)
        self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1] - 1) // 2)
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self.d_code = d_code
        self.n_labels = n_labels

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.fc2(self.relu(x))
        x = self.bn2(x) if self.bn2 is not None else x
        # print(x.shape)
        x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
        x = self.upsamp(x)
        x = self.relu(self.conv1(x))
        x = self.upsamp(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x

    def get_code(self, batch_size, device='cuda', return_labels=True, labels=None):
        if labels is None:  # sample labels
            labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
        code = pt.randn(batch_size, self.d_code, device=device)
        gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
        gen_one_hots.scatter_(1, labels, 1)
        code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
        # print(code.shape)
        if return_labels:
            return code, gen_one_hots
        else:
            return code


class ConvCondGenSVHN(nn.Module):
    def __init__(self, d_code, fc_spec, n_labels, nc_str, ks_str, use_sigmoid=False, batch_norm=True):
        super(ConvCondGenSVHN, self).__init__()
        self.nc = [int(k) for k in nc_str.split(',')] + [3]  # number of channels
        self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
        fc_spec = [int(k) for k in fc_spec.split(',')]
        assert len(self.nc) == 4 and len(self.ks) == 3
        self.hw = 4  # image height and width before upsampling
        self.reshape_size = self.nc[0] * self.hw ** 2
        self.fc1 = nn.Linear(d_code + n_labels, fc_spec[0])
        self.fc2 = nn.Linear(fc_spec[0], self.reshape_size)
        self.bn1 = nn.BatchNorm1d(fc_spec[0]) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(self.reshape_size) if batch_norm else None
        self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0] - 1) // 2)
        self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1] - 1) // 2)
        self.conv3 = nn.Conv2d(self.nc[2], self.nc[3], kernel_size=self.ks[2], stride=1, padding=(self.ks[2] - 1) // 2)
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self.d_code = d_code
        self.n_labels = n_labels

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.fc2(self.relu(x))
        x = self.bn2(x) if self.bn2 is not None else x
        # print(x.shape)
        x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
        x = self.upsamp(x)
        x = self.relu(self.conv1(x))
        x = self.upsamp(x)
        x = self.relu(self.conv2(x))
        x = self.upsamp(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x

    def get_code(self, batch_size, device, return_labels=True, labels=None):
        if labels is None:  # sample labels
            labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
        code = pt.randn(batch_size, self.d_code, device=device)
        gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
        gen_one_hots.scatter_(1, labels, 1)
        code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
        # print(code.shape)
        if return_labels:
            return code, gen_one_hots
        else:
            return code


class ConvGenSVHN(nn.Module):
    def __init__(self, d_code, fc_spec, n_labels, nc_str, ks_str, use_sigmoid=False, batch_norm=True):
        super(ConvGenSVHN, self).__init__()
        self.nc = [int(k) for k in nc_str.split(',')] + [3]  # number of channels
        self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
        assert len(self.nc) == 4 and len(self.ks) == 3
        self.hw = 4  # image height and width before upsampling
        self.reshape_size = self.nc[0] * self.hw ** 2
        fc_spec = [d_code] + [int(k) for k in fc_spec.split(',')] + [self.reshape_size]
        # print(fc_spec)
        self.fc1 = nn.Linear(fc_spec[0], fc_spec[1])
        self.bn1 = nn.BatchNorm1d(fc_spec[1]) if batch_norm else None
        self.fc2 = nn.Linear(fc_spec[1], fc_spec[2])
        self.bn2 = nn.BatchNorm1d(fc_spec[2]) if batch_norm else None
        if len(fc_spec) == 4:
            self.fc3 = nn.Linear(fc_spec[2], fc_spec[3])
            self.bn3 = nn.BatchNorm1d(fc_spec[3]) if batch_norm else None
        else:
            self.fc3, self.bn3 = None, None
        self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0] - 1) // 2)
        self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1] - 1) // 2)
        self.conv3 = nn.Conv2d(self.nc[2], self.nc[3], kernel_size=self.ks[2], stride=1, padding=(self.ks[2] - 1) // 2)
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self.d_code = d_code
        self.n_labels = n_labels

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.fc2(self.relu(x))
        x = self.bn2(x) if self.bn2 is not None else x

        x = self.fc3(self.relu(x)) if self.fc3 is not None else x
        x = self.bn3(x) if self.bn3 is not None else x

        x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
        x = self.upsamp(x)
        x = self.relu(self.conv1(x))
        x = self.upsamp(x)
        x = self.relu(self.conv2(x))
        x = self.upsamp(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x

    def get_code(self, batch_size, device):
        return pt.randn(batch_size, self.d_code, device=device)


class DeconvDecoder(nn.Module):

    def __init__(self, enc_size, ngf, n_channels=3, batch_norm=True, use_relu=True,
                 n_convs=4, n_extra_layers=0, first_deconv_kernel_size=4):
        super(DeconvDecoder, self).__init__()
        self.enc_size = enc_size
        nn_layers = OrderedDict()

        num_out_channels_first_conv = ngf * 8
        if n_convs < 4:
            num_out_channels_first_conv = ngf * 4

        # first deconv goes from the encoding size
        nn_layers[f"deconv_{len(nn_layers)}"] = nn.ConvTranspose2d(enc_size,
                                                                   num_out_channels_first_conv,
                                                                   first_deconv_kernel_size,
                                                                   1, 0, bias=False)
        if batch_norm:
            nn_layers[f"btn_{len(nn_layers)}"] = nn.BatchNorm2d(num_out_channels_first_conv)
        if use_relu:
            nn_layers[f"relu_{len(nn_layers)}"] = nn.ReLU(True)

        for i in range(n_convs - 4):
            self.create_deconv_block(nn_layers, ngf * 8, ngf * 8, batch_norm, use_relu)

        if n_convs >= 4:
            self.create_deconv_block(nn_layers, ngf * 8, ngf * 4, batch_norm, use_relu)

        self.create_deconv_block(nn_layers, ngf * 4, ngf * 2, batch_norm, use_relu)
        self.create_deconv_block(nn_layers, ngf * 2, ngf, batch_norm, use_relu)

        for i in range(n_extra_layers):
            self.create_conv_block(nn_layers, ngf, ngf, batch_norm, use_relu)

        self.create_deconv_block(nn_layers, ngf, n_channels, False, False)
        nn_layers[f"tanh_{len(nn_layers)}"] = nn.Tanh()

        self.net = nn.Sequential(nn_layers)

    @staticmethod
    def create_deconv_block(layers_dict, input_nc, output_nc, batch_norm=True, use_relu=True):
        layers_dict[f"deconv_{len(layers_dict)}"] = nn.ConvTranspose2d(input_nc, output_nc, 4, 2, 1,
                                                                       bias=False)
        if batch_norm:
            layers_dict[f"btnDeconv_{len(layers_dict)}"] = nn.BatchNorm2d(output_nc)
        if use_relu:
            layers_dict[f"reluDeconv_{len(layers_dict)}"] = nn.ReLU(True)

    @staticmethod
    def create_conv_block(layers_dict, input_nc, output_nc, batch_norm=True, use_relu=True):
        layers_dict[f"conv_{len(layers_dict)}"] = nn.Conv2d(input_nc, output_nc, 3, 1, 1, bias=False)
        if batch_norm:
            layers_dict[f"btnConv_{len(layers_dict)}"] = nn.BatchNorm2d(output_nc)
        if use_relu:
            layers_dict[f"reluConv_{len(layers_dict)}"] = nn.ReLU(True)

    def forward(self, x):
        # noinspection PyUnresolvedReferences
        # if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #   output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        # else:
        # output =
        return self.net(x)


class ResnetG(nn.Module):
    def __init__(self, enc_size, nc, ndf, image_size=32, adapt_filter_size=False,
                 use_conv_at_skip_conn=False, gen_output='tanh'):
        super(ResnetG, self).__init__()
        self.enc_size = enc_size
        self.ndf = ndf
        self.gen_output = gen_output

        if adapt_filter_size is True and use_conv_at_skip_conn is False:
            use_conv_at_skip_conn = True
            LOG.warning("WARNING: In ResnetG, setting use_conv_at_skip_conn to True because "
                        "adapt_filter_size is True.")

        n_upsample_blocks = int(log(image_size, 2)) - 2

        n_layers = n_upsample_blocks + 1
        filter_size_per_layer = [ndf] * n_layers
        if adapt_filter_size:
            for i in range(n_layers - 1, -1, -1):
                if i == n_layers - 1:
                    filter_size_per_layer[i] = ndf
                else:
                    filter_size_per_layer[i] = filter_size_per_layer[i + 1] * 2

        first_layer = nn.ConvTranspose2d(enc_size, filter_size_per_layer[0], 4, 1, 0, bias=False)
        nn.init.xavier_uniform_(first_layer.weight.data, 1.)
        last_layer = nn.Conv2d(filter_size_per_layer[-1], nc, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(last_layer.weight.data, 1.)

        nn_layers = OrderedDict()
        # first deconv goes from the z size
        nn_layers["firstConv"] = first_layer

        layer_number = 1
        for i in range(n_upsample_blocks):
            nn_layers[f"resblock_{i}"] = ResidualBlockG(filter_size_per_layer[layer_number - 1],
                                                        filter_size_per_layer[layer_number], stride=2,
                                                        use_conv_at_skip_conn=use_conv_at_skip_conn)
            layer_number += 1
        nn_layers["batchNorm"] = nn.BatchNorm2d(filter_size_per_layer[-1])
        nn_layers["relu"] = nn.ReLU()
        nn_layers["lastConv"] = last_layer
        if self.gen_output == 'tanh':
            nn_layers["tanh"] = nn.Tanh()

        self.net = nn.Sequential(nn_layers)

    def forward(self, x):
        return self.net(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, size=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        x = nnf.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode='nearest')
        return x

    def get_input_noise(self, batch_size):
        return torch.randn(batch_size, self.enc_size, 1, 1)


class ResidualBlockG(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_conv_at_skip_conn=False):
        super(ResidualBlockG, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        if use_conv_at_skip_conn:
            self.conv_bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.conv_bypass.weight.data, 1.)

        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
        )
        self.bypass = nn.Sequential()
        if stride != 1:
            if use_conv_at_skip_conn:
                self.bypass = nn.Sequential(self.conv_bypass, Upsample(scale_factor=2))
            else:
                self.bypass = Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)



############################### generative models to use ###############################
""" two types of generative models depending on the type of features in a given dataset """

class Generative_Model_homogeneous_data(nn.Module):

        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dataset):
            super(Generative_Model_homogeneous_data, self).__init__()

            self.input_size = input_size
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.output_size = output_size

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
            self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
            self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
            self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)

            self.dataset = dataset


        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(self.bn1(hidden))
            output = self.fc2(relu)
            output = self.relu(self.bn2(output))
            output = self.fc3(output)
            # output = self.sigmoid(output) # because we preprocess data such that each feature is [0,1]


            # if str(self.dataset) == 'epileptic':
            #     output = self.sigmoid(output) # because we preprocess data such that each feature is [0,1]
            # elif str(self.dataset) == 'isolet':
            #     output = self.sigmoid(output)

            return output


class Generative_Model_heterogeneous_data(nn.Module):

            def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, num_categorical_inputs, num_numerical_inputs):
                super(Generative_Model_heterogeneous_data, self).__init__()

                self.input_size = input_size
                self.hidden_size_1 = hidden_size_1
                self.hidden_size_2 = hidden_size_2
                self.output_size = output_size
                self.num_numerical_inputs = num_numerical_inputs
                self.num_categorical_inputs = num_categorical_inputs

                self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
                self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
                self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
                self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                hidden = self.fc1(x)
                relu = self.relu(self.bn1(hidden))
                output = self.fc2(relu)
                output = self.relu(self.bn2(output))
                output = self.fc3(output)

                output_numerical = self.relu(output[:, 0:self.num_numerical_inputs])  # these numerical values are non-negative
                # output_numerical = self.sigmoid(output_numerical) # because we preprocess data such that each feature is [0,1]
                output_categorical = self.sigmoid(output[:, self.num_numerical_inputs:])
                output_combined = torch.cat((output_numerical, output_categorical), 1)

                return output_combined

############################### end of generative models ###############################