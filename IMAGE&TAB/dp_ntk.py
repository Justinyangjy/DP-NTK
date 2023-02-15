import argparse
import os
import random
from collections import Counter

import numpy as np
import torch as pt

from all_aux_tab import data_loading
from data_loading import load_cifar10
from dp_ntk_gen_step import gen_step
from dp_ntk_mean_emb1 import calc_mean_emb1
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
    parser.add_argument('--data', type=str, default='adult', help='cifar10 or tabular: adult, census, cervical, '
                                                                    'isolet,credit, epileptic, intrusion, covtype')
    parser.add_argument('--id', default="None", help='custom description of the set-up')

    # OPTIMIZATION
    parser.add_argument('--emb_batch-size', '-bs', type=int, default=100)
    parser.add_argument('--test-batch-size', '-tbs', type=int, default=100)
    parser.add_argument('--gen-batch-size', '-gbs', type=int, default=800)
    parser.add_argument('--n_iter', type=int, default=20_00)
    parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='learning rate decay factor')
    parser.add_argument('--scheduler-interval', type=int, default=500,
                        help='reduce lr after n steps')

    # MODEL DEFINITION
    parser.add_argument('--d-code', '-dcode', type=int, default=100, help='random code dimensionality')
    parser.add_argument('--gen-spec', type=str, default="200,200")
    parser.add_argument('--model-ntk', default="fc_1l")
    parser.add_argument('--ntk-width', type=int, default=800, help='width of NTK for apprixmate mmd')
    parser.add_argument('--ntk-width-2', type=int, default=1000, help='width of NTK for apprixmate mmd 2nd layer')

    # DP SPEC
    parser.add_argument('--tgt-eps', type=float, default=None, help='privacy parameter - finds noise')
    parser.add_argument('--tgt-delta', type=float, default=1e-5, help='privacy parameter - finds noise')

    parser.add_argument('--tab_classifiers', nargs='+', type=int, help='list of integers',
                        default=[3, 4])  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

    ar = parser.parse_args()
    preprocess_args(ar)
    print(ar)
    return ar


def preprocess_args(ar):
    if ar.log_dir is None:
        assert ar.data in ['cifar10', "adult", "census", "cervical", "isolet", "credit", "epileptic", "intrusion",
                           "covtype"]
        ar.log_dir = ar.base_log_dir + ar.data + '/'

    os.makedirs(ar.log_dir, exist_ok=True)

    if ar.seed is None:
        ar.seed = np.random.randint(0, 1000)

    if ar.tgt_eps is None:
        ar.is_private = 0
    else:
        ar.is_private = 1


def main():
    ar = get_args()
    random.seed(ar.seed)
    pt.manual_seed(ar.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device is', device)

    # data

    """ load Cifar10 or Tabular data"""
    labels_distribution, test_data, y_test = None, None, None
    if ar.data == 'cifar10':
        train_loader, n_classes = load_cifar10(image_size=32, dataroot=ar.log_dir, use_autoencoder=False,
                                               batch_size=100, n_workers=2, labeled=True,
                                               test_set=False, scale_to_range=False)
        input_dim = 32 * 32 * 3
        n_train = 50_000
    else:
        train_data, test_data, labels, y_test, n_classes, num_categorical_inputs, num_numerical_inputs = data_loading(
            ar.data)
        print("Train tab data: ", train_data.shape)
        tensor_x = torch.stack([torch.Tensor(i) for i in train_data])  # transform to torch tensors
        tensor_y = torch.stack([torch.Tensor(np.array([i])) for i in labels])
        train_dataset = pt.utils.data.TensorDataset(tensor_x, tensor_y)  # create your dataset
        train_loader = pt.utils.data.DataLoader(train_dataset, batch_size=ar.emb_batch_size)  # create your dataloader

        # one-hot encoding of labels.
        n_train, input_dim = train_data.shape
        labels_counts = list(Counter(labels).values())
        labels_distribution = np.array(labels_counts) / sum(labels_counts)
        n_classes = len(labels_distribution)
        print("labels distr", labels_distribution)

    if ar.model_ntk == "fc_1l":
        model_ntk = NTK(input_size=input_dim, hidden_size_1=ar.ntk_width, output_size=n_classes)
    elif ar.model_ntk == "fc_2l":
        model_ntk = NTK_TL(input_size=input_dim, hidden_size_1=ar.ntk_width, hidden_size_2=ar.ntk_width_2,
                           output_size=n_classes)  # output=n_classes
    elif ar.model_ntk == "cnn_2l":
        if ar.data == 'cifar10':
            model_ntk = CNTK(ar.ntk_width)
        else:
            model_ntk = CNTK_1D(input_dim, ar.ntk_width, ar.ntk_width_2, output_size=n_classes)
    elif ar.model_ntk == "cnn_1l":
        if ar.data == 'cifar10':
            model_ntk = CNTK_2L(ar.ntk_width, ar.ntk_width_2)
        else:
            model_ntk = CNTK_1D_1L(input_dim, ar.ntk_width, ar.ntk_width_2, output_size=n_classes)

    model_ntk.to(device)
    model_ntk.eval()

    print('computing mean embedding of true data')
    calc_mean_emb1(model_ntk, ar, device, train_loader, n_classes)
    print('generator step')
    acc = gen_step(model_ntk, ar, device, n_train, labels_distribution, test_data, y_test)

    return acc


if __name__ == '__main__':
    acc = main()
