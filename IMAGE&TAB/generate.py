import os
import random
import argparse
import matplotlib
import matplotlib.pyplot as plt
import tkinter

matplotlib.use('tkagg')
import numpy as np
import torch
import torch as pt
import torch.optim as optim
# from models.resnet9_ntk import ResNet
from autodp import privacy_calibrator
from torch.optim.lr_scheduler import StepLR
from torchvision import utils as vutils

from fid_eval import get_fid_scores
# from models_gen import FCCondGen, ConvCondGen
from models.generators import ResnetG, ConvCondGen
from models.ntk import *
from synth_data_benchmark import test_gen_data
from util import plot_mnist_batch, log_final_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device is', device)

args = argparse.ArgumentParser()

args.add_argument("--data", default="celeba")
args.add_argument("--d_code", type=int, default=201)
args.add_argument("--log_dir", default="/home/kamil/Dropbox/Current_research/MMD_NTK/logs/")
args.add_argument("--gen_batch_size", default=100, type=int)
args.add_argument("--batch_size", default=100, type=int)
args.add_argument("--id", default="None")


ar = args.parse_args()




ar.data = "celeba"

""" setup """
if ar.data == 'cifar10':
    iput_dim = 32 * 32 * 3
    image_size = 32
    n_data = 50_000
elif ar.data == 'celeba':
    iput_dim = 32 * 32 * 3
    image_size = 32
    n_data = 202_599
else:
    input_dim = 784
    image_size = 28
    n_data = 60_000
n_classes = 1

if ar.data == 'cifar10':
    model = ResnetG(ar.d_code + n_classes, nc=3, ndf=64, image_size=32,
                    adapt_filter_size=True,
                    use_conv_at_skip_conn=False).to(device)
elif ar.data == 'celeba':
    model = ResnetG(ar.d_code + n_classes, nc=3, ndf=64, image_size=32,
                    adapt_filter_size=True,
                    use_conv_at_skip_conn=False).to(device)
else:
    model = ConvCondGen(ar.d_code, ar.gen_spec, n_classes, '16,8', '5,5').to(device)



def get_code_fun(device, n_labels, d_code):
    def get_code(batch_size, labels=None):
        return_labels = False
        if labels is None:  # sample labels
            return_labels = True
            labels = pt.randint(n_labels, (batch_size, 1), device=device)
        code = pt.randn(batch_size, d_code, device=device)
        gen_one_hots = pt.zeros(batch_size, n_labels, device=device)
        gen_one_hots.scatter_(1, labels, 1)
        code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)[:, :, None, None]
        if return_labels:
            return code, gen_one_hots
        else:
            return code

    return get_code

def synthesize_data_with_uniform_labels(gen, device, code_fun, gen_batch_size=1000, n_data=60000,n_labels=10):
    gen.eval()
    if n_data % gen_batch_size != 0:
        assert n_data % 100 == 0
        gen_batch_size = n_data // 100
    assert gen_batch_size % n_labels == 0
    n_iterations = n_data // gen_batch_size

    data_list = []
    ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
    labels_list = [ordered_labels] * n_iterations

    with pt.no_grad():
        for idx in range(n_iterations):
            gen_code = code_fun(gen_batch_size, labels=ordered_labels)
            gen_samples = gen(gen_code)
            data_list.append(gen_samples)
    return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()

def log_step(gen, fixed_noise, exp_name, ar):
    #print(f'Train step: {step} \tLoss: {loss.item():.6f}')
    log_dir = ar.log_dir
    with pt.no_grad():
        fake = gen(fixed_noise).detach()
    img_dir = os.path.join(log_dir, 'images/')
    os.makedirs(img_dir, exist_ok=True)
    print(fake.data.shape)
    for i in range(28):
        img_name = img_dir+ar.data+str(i)+".png"
        vutils.save_image(fake.data[i*10:(i+1)*10], img_name, normalize=True, nrow=5)

if ar.data == 'cifar10' or ar.data == 'celeba':

    code_fun = get_code_fun(device, n_classes, ar.d_code)
    # save trained model and data
    path_data  = ar.log_dir + "generated/good/"+'celeba_iter_20000-dcode_201-fc_2l_1500_200-batch_1000-0.01-eps_10-priv_1gen_data.npz'
    path_gen = ar.log_dir + "generated/good/"+'celeba_iter_20000-dcode_201-fc_2l_1500_200-batch_1000-0.01-eps_10-priv_1gen.pt'

    model.load_state_dict(torch.load(path_gen))
    syn_data_size = 50000
    syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, code_fun, ar.gen_batch_size, syn_data_size, n_classes)
    path_generated="../logs/generated/my_gen_data.npz"
    np.savez(path_generated, x=syn_data, y=syn_labels)

    # draw
    fixed_labels = pt.repeat_interleave(pt.arange(1, device=device), 1000)
    print(fixed_labels.shape)
    fixed_noise = code_fun(batch_size=1000, labels=fixed_labels[:, None])
    log_step(model, fixed_noise, 'test', ar)


    print(syn_data.shape)

    fig = plt.figure()
    plt.imshow(syn_data[0].transpose(1, 2, 0))
    plt.savefig("test.png")

    path = path_generated


    # fig

    fid_score = get_fid_scores(path, ar.data, device, syn_data_size,image_size, center_crop_size=32, use_autoencoder=False,base_data_dir='../data', batch_size=100)
    print(f'fid={fid_score}')




    #np.save(os.path.join(ar.log_dir, 'fid.npy'), fid_score)
# else:
#     """plot generation """
#     log_gen_data(model, device, ar.n_iter, n_classes, ar.log_dir)
#
#     """evaluate the model"""
#     pt.save(model.state_dict(), ar.log_dir + 'gen.pt')
#     syn_data, syn_labels = synthesize_mnist_with_uniform_labels(model, device)
#     np.savez(ar.log_dir + 'synthetic_mnist', data=syn_data, labels=syn_labels)
#     final_score = test_gen_data(ar.log_dir, ar.data, data_base_dir="", subsample=0.1, custom_keys="logistic_reg",
#                                 data_from_torch=True)
#     log_final_score(ar.log_dir, final_score)
#
#     final_score = test_gen_data(ar.log_dir, ar.data, data_base_dir="", subsample=0.1, custom_keys="mlp",
#                                 data_from_torch=True)
#     log_final_score(ar.log_dir, final_score)


