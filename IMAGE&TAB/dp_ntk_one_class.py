import argparse
import os
import random

import numpy as np
import torch as pt
from torchvision.models import resnet18

from dp_ntk_gen_step_one_class import gen_step
from dp_ntk_mean_emb1_one_class import calc_mean_emb1
from models.ntk import *


def get_args():
    parser = argparse.ArgumentParser()

    # BASICS
    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--log-interval', type=int, default=100, help='print updates after n steps')
    parser.add_argument('--base-log-dir', type=str, default='res/',
                        help='path where logs for all runs are stored')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='override save path. constructed if None')
    parser.add_argument('--data', type=str, default='cifar10', help='cifar10 or celeba')
    parser.add_argument('--id', default="None", help='custom description of the set-up')

    # OPTIMIZATION
    parser.add_argument('--batch-size', '-bs', type=int, default=125)
    parser.add_argument('--test-batch-size', '-tbs', type=int, default=100)
    parser.add_argument('--gen-batch-size', '-gbs', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=1_000)
    parser.add_argument('--lr', '-lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='learning rate decay factor')
    parser.add_argument('--scheduler-interval', type=int, default=1000,
                        help='reduce lr after n steps')
    # MODEL DEFINITION
    parser.add_argument('--d-code', '-dcode', type=int, default=100, help='random code dimensionality')
    parser.add_argument('--gen-spec', type=str, default="200,200")
    parser.add_argument('--model-ntk', default="cnn2d_1l")
    parser.add_argument('--ntk-width', type=int, default=20, help='width of NTK for apprixmate mmd')
    parser.add_argument('--ntk-width-2', type=int, default=100, help='width of NTK for apprixmate mmd 2nd layer')


    # DP SPEC
    parser.add_argument('--is-private', type=int, default=1)
    parser.add_argument('--tgt-eps', type=float, default=None, help='privacy parameter - finds noise')
    parser.add_argument('--tgt-delta', type=float, default=1e-5, help='privacy parameter - finds noise')

    ar = parser.parse_args()

    preprocess_args(ar)
    return ar


def preprocess_args(ar):
    if ar.log_dir is None:
        assert ar.data in ['cifar10', 'celeba']
        ar.log_dir = ar.base_log_dir + ar.data + '/'

    os.makedirs(ar.log_dir, exist_ok=True)

    if ar.seed is None:
        ar.seed = np.random.randint(0, 1000)


def main(data=None):
    ar = get_args()
    if data is not None:
        ar.data = data
    print(ar)
    random.seed(ar.seed)
    pt.manual_seed(ar.seed)

    """ load celeba or cifar10 """
    if ar.data == 'cifar10':
        input_dim = 32 * 32 * 3
        n_data = 50_000
    else:
        input_dim = 32 * 32 * 3
        n_data = 202_599

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device is', device)

    # model_ntk = NTK_TL(input_size=input_dim, hidden_size_1=500, hidden_size_2=500, output_size=10)
    # model_ntk = resnet18()
    # model_ntk.fc = torch.nn.Linear(512, 10, bias=False)
    if ar.model_ntk == "fc_1l":
        model_ntk = NTK(input_size=input_dim, hidden_size_1=ar.ntk_width, output_size=1)
    elif ar.model_ntk == "fc_2l":
        model_ntk = NTK_TL(input_size=input_dim, hidden_size_1=ar.ntk_width, hidden_size_2=ar.ntk_width_2,output_size=1)  # output=n_classes
    elif ar.model_ntk == "cnn2d_1l":
        model_ntk = CNTK(ar.ntk_width)
    elif ar.model_ntk == "cnn2d_2l":
        model_ntk = CNTK_2L(ar.ntk_width, ar.ntk_width_2)
    # model_ntk = ResNet()
    # model_ntk = model_ntk.net
    # model_ntk_pretrain = Net_eNTK_pretrain()
    # model_ntk = get_ffcv_model(device, num_class=1)
    # model_ntk = Net_eNTK()
    # model_ntk = pt.load('model_cNTK.pth')
    # model_ntk = pt.load('model_ResNet9_imagenet.pth')
    # model_ntk_pretrain.load_state_dict(pt.load('model_cNTK_cifar.pth', map_location=device))
    # model_ntk.load_state_dict(pt.load('model_cNTK_cifar.pth', map_location=device))

    # model_ntk.fc1 = nn.Linear(4096, 1)
    # model_ntk.load_parameters(path='model_cNTK.pth')
    # model_ntk[-2] = torch.nn.Linear(128, 1, bias=False)
    model_ntk.to(device)
    model_ntk.eval()
    # print(model_ntk)
    # print(model_ntk)
    # model_ntk_pretrain.to(device)
    # output_weights = model_ntk_pretrain.fc1.weight
    # print(output_weights[0,:])
    # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[0,:])
    # print(model_ntk.fc1.weight)

    print('computing mean embedding of true data')
    calc_mean_emb1(model_ntk, ar, device)
    print('generator step')
    acc = gen_step(model_ntk, ar, device)

    return acc


if __name__ == '__main__':
    main()