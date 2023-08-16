# from models.generators import ResnetG
# from synth_data_2d import plot_data
# from models.resnet9_ntk import ResNet
import random

import torch as pt

# from util import plot_mnist_batch, log_final_score
from data_loading import load_cifar10, get_mnist_dataloaders, load_dataset
from models.ntk import *


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
        train_loader, n_classes = load_cifar10(image_size=32, dataroot='./data/', use_autoencoder=True,batch_size=ar.batch_size, n_workers=2, labeled=True, test_set=False, scale_to_range=False)
        input_dim = 32 * 32 * 3
        n_data = 50_000
        n_classes = 1
    else:
        train_loader, _ = load_dataset('celeba', image_size=32, center_crop_size=32, dataroot='./data/',
                                       use_autoencoder=True, batch_size=100,
                                       n_workers=2, labeled=False, test_set=False)
        input_dim = 32 * 32 * 3
        n_data = 202_599
        n_classes = 1

    """ initialize the variables"""

    mean_v_samp = torch.Tensor([]).to(device)
    for p in model_ntk.parameters():
        mean_v_samp = torch.cat((mean_v_samp, p.flatten()))
    d = len(mean_v_samp) - 1
    mean_emb1 = torch.zeros(d, device=device)
    print('Feature Length:', d)
    n_c = n_data

    for data, labels in train_loader:
        data = data.to(device)
        for i in range(data.shape[0]):
            """ manually set the weight if needed """
            # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[y_train[i],:][None,:])

            mean_v_samp = torch.Tensor([]).to(device)  # sample mean vector init
            f_x = model_ntk(data[i][None, :, :, :])

            """ get NTK features """
            f_idx_grad = torch.autograd.grad(f_x, model_ntk.parameters(),
                                             grad_outputs=f_x.data.new(f_x.shape).fill_(1))
            for g in f_idx_grad:
                mean_v_samp = torch.cat((mean_v_samp, g.flatten()))
            mean_v_samp = mean_v_samp[:-1]  # Removing last feature as gradient is always 1.

            """ normalize the sample mean vector """
            if ar.is_private:
                mean_emb1 += mean_v_samp / torch.norm(mean_v_samp)
            else:
                mean_emb1 += mean_v_samp

    """ average by class count """
    mean_emb1 = torch.div(mean_emb1, n_c)
    print("This is the shape for dp-ntk mean_emb1: ", mean_emb1.shape)

    """ save model for downstream task """
    torch.save(mean_emb1, ar.log_dir + 'mean_emb1_1_' + str(d) + '.pth')

    torch.save(model_ntk.state_dict(), ar.log_dir + 'model_' + str(d) + '.pth')
