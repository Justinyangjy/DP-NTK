import argparse
import os
import random

import numpy as np
import torch as pt

from dp_ntk_gen import gen_step
from dp_ntk_mean_emb1 import calc_mean_emb1
from models.ntk import *


def get_args():
    parser = argparse.ArgumentParser()

    # BASICS
    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='print updates after n steps')
    parser.add_argument('--base-log-dir', type=str, default='res/',
                        help='path where logs for all runs are stored')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='override save path. constructed if None')
    parser.add_argument('--data', type=str, default='dmnist', help='cifar10, dmnist or fmnist')
    parser.add_argument('--calc-data', type=bool, default=True,
                        help='whether or not to calculate the true data mean embedding with DP levels 0.2, 1, and 10')

    # OPTIMIZATION
    parser.add_argument('--batch-size', '-bs', type=int, default=500)
    parser.add_argument('--test-batch-size', '-tbs', type=int, default=100)
    parser.add_argument('--gen-batch-size', '-gbs', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=2000)
    parser.add_argument('--lr', '-lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='learning rate decay factor')
    parser.add_argument('--scheduler-interval', type=int, default=1500,
                        help='reduce lr after n steps')

    # MODEL DEFINITION
    parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')
    parser.add_argument('--gen-spec', type=str, default="200,200")
    parser.add_argument('--ntk-width', type=int, default=500, help='width of NTK for apprixmate mmd')

    # DP SPEC
    parser.add_argument('--tgt-eps', type=float, default=None, help='privacy parameter - finds noise')
    parser.add_argument('--tgt-delta', type=float, default=1e-5, help='privacy parameter - finds noise')

    ar = parser.parse_args()

    preprocess_args(ar)
    return ar


def preprocess_args(ar):
    if ar.log_dir is None:
        assert ar.data in ['cifar10', 'dmnist', 'fmnist']
        ar.log_dir = ar.base_log_dir + ar.data + '/'

    os.makedirs(ar.log_dir, exist_ok=True)

    if ar.seed is None:
        ar.seed = np.random.randint(0, 1000)


def main():
    ar = get_args()
    random.seed(ar.seed)
    pt.manual_seed(ar.seed)

    """ load MNIST, FashionMNIST or cifar10 """
    if ar.data == 'cifar10':
        input_dim = 32 * 32 * 3
        n_data = 50_000
        n_classes = 10
    else:
        input_dim = 784
        n_data = 60_000
        n_classes = 10
        eval_func = None

    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print('device is', device)

    # model_ntk = NTK_TL(input_size=input_dim, hidden_size_1=10, hidden_size_2=ar.ntk_width, output_size=n_classes)
    # model_ntk = ResNet(input_size=input_dim, hidden_size_1=ar.ntk_width, output_size=n_classes)
    model_ntk = NTK(input_size=input_dim, hidden_size_1=ar.ntk_width, output_size=n_classes)
    # model_ntk = NTK(input_size=input_dim, hidden_size_1=ar.ntk_width, output_size=n_classes)
    # model_ntk = CNTK()
    # model_ntk = ResNet()
    # model_ntk = model_ntk.net
    # model_ntk_pretrain = Net_eNTK_pretrain()
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

    if ar.calc_data:
        print('computing mean embedding of true data')
        calc_mean_emb1(model_ntk, ar, device)
    else:
        print('skipping the computing mean embedding of true data')
    print('generator step')
    gen_step(model_ntk, ar, device)


if __name__ == '__main__':
    main()