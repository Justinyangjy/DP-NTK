import argparse
# from models.generators import ResnetG
# from synth_data_2d import plot_data
# from models.resnet9_ntk import ResNet
import os
import random

import numpy as np
import torch as pt

# from util import plot_mnist_batch, log_final_score
from data_loading import load_cifar10, get_mnist_dataloaders
from models.ntk import *
from autodp import privacy_calibrator


def synthesize_mnist_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
    gen.eval()
    assert n_data % gen_batch_size == 0
    assert gen_batch_size % n_labels == 0
    n_iterations = n_data // gen_batch_size

    data_list = []
    ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
    labels_list = [ordered_labels] * n_iterations

    with pt.no_grad():
        for idx in range(n_iterations):
            gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
            gen_samples = gen(gen_code)
            data_list.append(gen_samples)
    return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()


def calc_mean_emb1(model_ntk, ar, device):
    random.seed(ar.seed)
    pt.manual_seed(ar.seed)

    """ load MNIST, FashionMNIST or cifar10 """
    if ar.data == 'cifar10':
        train_loader, n_classes = load_cifar10(image_size=32, dataroot='./data/', use_autoencoder=False,
                                               batch_size=ar.batch_size, n_workers=2, labeled=True,
                                               test_set=False, scale_to_range=False)
        input_dim = 32 * 32 * 3
        n_data = 50_000
    else:
        train_loader, test_loader, trn_data, tst_data = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size,
                                                                              use_cuda=True,
                                                                              dataset=ar.data, normalize=False,
                                                                              return_datasets=True)
        input_dim = 784
        n_data = 60_000
        n_classes = 10
        eval_func = None

    """ initialize the variables"""

    mean_v_samp = pt.Tensor([]).to(device)
    for p in model_ntk.parameters():
        mean_v_samp = pt.cat((mean_v_samp, p.flatten()))
    d = len(mean_v_samp)
    mean_emb1 = pt.zeros((d, n_classes), device=device)
    print('Feature Length:', d)

    for data, labels in train_loader:
        data, y_train = data.to(device), labels.to(device)
        for i in range(data.shape[0]):
            """ manually set the weight if needed """
            # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[y_train[i],:][None,:])

            mean_v_samp = pt.Tensor([]).to(device)  # sample mean vector init
            if ar.data == 'cifar10':
                f_x = model_ntk(data[i][None, :, :, :])  # 1 input, dimensions need tweaking
            else:
                f_x = model_ntk(data[i])

            """ get NTK features """
            f_idx_grad = pt.autograd.grad(f_x, model_ntk.parameters(),
                                             grad_outputs=f_x.data.new(f_x.shape).fill_(1))
            for g in f_idx_grad:
                mean_v_samp = pt.cat((mean_v_samp, g.flatten()))
            # mean_v_samp = mean_v_samp[:-1]

            """ normalize the sample mean vector """
            mean_emb1[:, y_train[i]] += mean_v_samp / pt.linalg.vector_norm(mean_v_samp)

    """ average by class count """
    mean_emb1 = pt.div(mean_emb1, n_data)
    print("This is the shape for dp-mint mean_emb1: ", mean_emb1.shape)

    """ save mean_emb1 """
    print(ar.save_dir + f'mean_emb1_{d}_{ar.seed}.pth')
    pt.save(mean_emb1, ar.save_dir + f'mean_emb1_{d}_{ar.seed}.pth')

    """ adding DP noise to sensitive data """
    if ar.tgt_eps is not None:
        noise_mean_emb1 = mean_emb1
        privacy_param = privacy_calibrator.gaussian_mech(ar.tgt_eps, ar.tgt_delta, k=1)
        print(f'eps,delta = ({ar.tgt_eps},{ar.tgt_delta}) ==> Noise level sigma=', privacy_param['sigma'])
        noise = pt.randn(mean_emb1.shape[0], mean_emb1.shape[1], device=device)
        std = 2 * privacy_param['sigma'] / n_data
        noise = noise * std
        noise_mean_emb1 += noise
        pt.save(noise_mean_emb1, ar.save_dir + f'mean_emb1_{d}_{ar.seed}_{ar.tgt_eps}.pth')

    """ save model """
    pt.save(model_ntk.state_dict(), ar.save_dir + f'model_{d}_{ar.seed}.pth')
