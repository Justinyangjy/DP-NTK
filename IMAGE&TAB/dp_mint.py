import argparse
import os
import random
from collections import Counter

import numpy as np
import torch as pt
#from torchvision.models import resnet18, ResNet18_Weights

#from dp_mepf_train_gen import dp_mepf
from dp_mint_gen_step import gen_step
from dp_mint_mean_emb1 import calc_mean_emb1
from models.ntk import *

from data_loading import load_cifar10, get_mnist_dataloaders
from all_aux_tab import data_loading


def get_args():
    parser = argparse.ArgumentParser()

    # BASICS
    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='print updates after n steps')
    parser.add_argument('--base-log-dir', type=str, default='res/',
                        help='path where logs for all runs are stored')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='override save path. constructed if None')
    parser.add_argument('--data', type=str, default='celeba', help='cifar10, dmnist or fmnist')
    parser.add_argument('--id', default="None", help='custom description of the set-up')

    # OPTIMIZATION
    parser.add_argument('--emb_batch-size', '-bs', type=int, default=100)
    parser.add_argument('--test-batch-size', '-tbs', type=int, default=100)
    parser.add_argument('--gen-batch-size', '-gbs', type=int, default=800)
    parser.add_argument('--n_iter', type=int, default=20_000)
    parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='learning rate decay factor')
    parser.add_argument('--scheduler-interval', type=int, default=500,
                        help='reduce lr after n steps')

    # MODEL DEFINITION
    parser.add_argument('--d-code', '-dcode', type=int, default=100, help='random code dimensionality')
    parser.add_argument('--gen-spec', type=str, default="200,200")
    parser.add_argument('--which-feat', type=str, default="ntk", help='ntk, pf or both')
    parser.add_argument('--model-ntk', default="fc_1l")
    parser.add_argument('--ntk-width', type=int, default=800, help='width of NTK for apprixmate mmd')
    parser.add_argument('--ntk-width-2', type=int, default=1000, help='width of NTK for apprixmate mmd 2nd layer')

    # DP SPEC
    parser.add_argument('--tgt-eps', type=float, default=1, help='privacy parameter - finds noise')
    parser.add_argument('--tgt-delta', type=float, default=1e-5, help='privacy parameter - finds noise')
    parser.add_argument('--is-private', type=int, default=1)

    parser.add_argument('--tab_classifiers', nargs='+', type=int, help='list of integers', default=[3, 4]) #0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

    ar = parser.parse_args()
    print(ar)
    preprocess_args(ar)
    return ar


def preprocess_args(ar):
    if ar.log_dir is None:
        assert ar.data in ['cifar10', 'dmnist', 'fmnist']
        ar.log_dir = ar.base_log_dir + ar.data + '/'

    os.makedirs(ar.log_dir, exist_ok=True)

    if ar.seed is None:
        ar.seed = np.random.randint(0, 1000)


def main(data=None):
    ar = get_args()
    if data is not None:
        ar.data =data
    random.seed(ar.seed)
    pt.manual_seed(ar.seed)

    """ load MNIST, FashionMNIST or cifar10 """

    # if ar.data == 'cifar10':
    #     input_dim = 32 * 32 * 3
    #     n_data = 50_000
    #     n_classes = 10
    # elif "mnist" in ar.data:
    #     input_dim = 784
    #     n_data = 60_000
    #     n_classes = 10
    #     eval_func = None
    # else:
    #     input_dim =14

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device is', device)


    # data

    """ load MNIST, FashionMNIST or cifar10 """
    labels_distribution, test_data, y_test = None, None, None
    if ar.data == 'cifar10':
        train_loader, n_classes, test_loader = load_cifar10(image_size=32, dataroot=ar.log_dir, use_autoencoder=False, batch_size=ar.batch_size, n_workers=2, labeled=True, test_set=False, scale_to_range=False)
        input_dim = 32 * 32 * 3
        n_train = 50_000
    elif ar.data == "mnist":
        train_loader, test_loader, trn_data, tst_data = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size,use_cuda=True,dataset=ar.data, normalize=False,return_datasets=True)
        input_dim = 784
        n_train = 60_000
        n_classes = 10
        eval_func = None
    else:
        train_data, test_data, labels, y_test, n_classes, num_categorical_inputs, num_numerical_inputs = data_loading(ar.data)
        print("Train tab data: ", train_data.shape)
        tensor_x = torch.stack([torch.Tensor(i) for i in train_data])  # transform to torch tensors
        tensor_y = torch.stack([torch.Tensor(np.array([i])) for i in labels])
        train_dataset = pt.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset
        train_loader = pt.utils.data.DataLoader(train_dataset, batch_size=ar.emb_batch_size)  # create your dataloader

        # test_tensor_x = torch.stack([torch.Tensor(i) for i in train_data])  # transform to torch tensors
        # test_tensor_y = torch.stack([torch.Tensor(np.array([i])) for i in y_test])
        #
        # test_dataset = pt.utils.data.TensorDataset(test_tensor_x, test_tensor_y)  # create your datset
        # test_loader = pt.utils.data.DataLoader(test_dataset, batch_size=ar.batch_size)  # create your dataloader

        # train_data, test_data, labels, y_test = pt.Tensor(train_loader), pt.Tensor(test_data), pt.LongTensor(labels), pt.LongTensor(y_test)

        # one-hot encoding of labels.
        n_train, input_dim = train_data.shape
        labels_counts = list(Counter(labels).values())
        labels_distribution = np.array(labels_counts)/sum(labels_counts)
        n_classes=len(labels_distribution)
        print("labels distr", labels_distribution)


    # net

    # model_ntk = ResNet(input_size=input_dim, hidden_size_1=ar.ntk_width, output_size=n_classes)
    # model_ntk = NTK(input_size=input_dim, hidden_size_1=9000, output_size=1)
    if ar.model_ntk=="fc_1l":
        model_ntk = NTK(input_size=input_dim, hidden_size_1=ar.ntk_width, output_size=1)
    elif ar.model_ntk=="fc_2l":
        model_ntk = NTK_TL(input_size=input_dim, hidden_size_1=ar.ntk_width, hidden_size_2=ar.ntk_width_2, output_size=1) #output=n_classes

    #model_ntk = CNTK()
    elif ar.model_ntk == "cnn_2l":
        model_ntk = CNTK_1D(input_dim, ar.ntk_width, ar.ntk_width_2, output_size=1)
    elif ar.model_ntk == "cnn_1l":
        model_ntk = CNTK_1D_1L(input_dim, ar.ntk_width, ar.ntk_width_2, output_size=1)
    # model_ntk = ResNet()
    # model_ntk = model_ntk.net
    # model_ntk_pretrain = Net_eNTK_pretrain()
    # model_ntk = get_ffcv_model(device, num_class=1000)
    # model_ntk = resnet18(ResNet18_Weights.IMAGENET1K_V1)
    # model_ntk.fc = torch.nn.Linear(512, 1, bias=False)
    # model_ntk = Net_eNTK()
    # model_ntk = pt.load('model_cNTK.pth')
    # model_ntk = pt.load('model_ResNet9.pth')
    # model_ntk_pretrain.load_state_dict(pt.load('model_cNTK_cifar.pth', map_location=device))
    # model_ntk.load_state_dict(pt.load('model_cNTK_cifar.pth', map_location=device))

    # model_ntk.fc1 = nn.Linear(4096, 1)
    # model_ntk.load_parameters(path='model_cNTK.pth')
    model_ntk.to(device)
    model_ntk.eval()
    # model_ntk_pretrain.to(device)
    # output_weights = model_ntk_pretrain.fc1.weight
    # print(output_weights[0,:])
    # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[0,:])
    # print(model_ntk.fc1.weight)

    if ar.which_feat != 'pf':
        print('computing mean embedding of true data')
        calc_mean_emb1(model_ntk, ar, device, train_loader, n_classes)
        print('generator step')
        acc = gen_step(model_ntk, ar, device, n_train, labels_distribution, test_data, y_test)
    else:
        dp_mepf(ar)

    return acc


if __name__ == '__main__':
    acc = main()
