import os
import random

import numpy as np
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


def log_gen_data(gen, device, epoch, n_labels, log_dir):
    ordered_labels = pt.repeat_interleave(pt.arange(n_labels), n_labels)[:, None].to(device)
    gen_code, _ = gen.get_code(100, device, labels=ordered_labels)
    gen_samples = gen(gen_code).detach()

    plot_samples = gen_samples[:100, ...].cpu().numpy()
    plot_mnist_batch(plot_samples, 10, n_labels, log_dir + f'samples_ep{epoch}', denorm=False)


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


def log_step(step, loss, gen, fixed_noise, exp_name, ar):
    print(f'Train step: {step} \tLoss: {loss.item():.6f}')
    log_dir = ar.log_dir
    with pt.no_grad():
        fake = gen(fixed_noise).detach()
    img_dir = os.path.join(log_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    if ar.id == "None":
        img_path = os.path.join(img_dir, f"{exp_name.replace('/', '_', 999)}_it_{step + 1}.png")
    else:
        img_path = os.path.join(img_dir, f"{ar.id}.png") #no step

    vutils.save_image(fake.data[:100], img_path, normalize=True, nrow=10)


def synthesize_data_with_uniform_labels(gen, device, code_fun, gen_batch_size=1000, n_data=60000,
                                        n_labels=10):
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


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def gen_step(model_ntk, ar, device):
    random.seed(ar.seed)
    pt.manual_seed(ar.seed)
    # reg = [0.001, 0.0001, 0.0001]  # [R, G, B]
    # reg = [1, 1, 1]
    # reg = torch.Tensor(reg).to(device)
    # reg = 1e-6
    print("reg:", ar.reg)

    # reg = 0.0001

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

    print('data: ', ar.data)
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

    optimizer = optim.Adam(model.parameters(), lr=ar.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
    training_loss_per_epoch = np.zeros(ar.n_iter)

    """ initialize the variables """
    mean_v_samp = torch.Tensor([]).to(device)
    for p in model_ntk.parameters():
        mean_v_samp = torch.cat((mean_v_samp, p.flatten()))
    d = len(mean_v_samp) - 1
    print('Feature Length:', d)
    mean_emb1_1 = pt.load(ar.log_dir + 'mean_emb1_1_' + str(d) + '.pth')
    mean_emb1_2 = pt.load(ar.log_dir + 'mean_emb1_2_' + ar.data + '.pth')
    model_ntk.load_state_dict(pt.load(ar.log_dir + 'model_' + str(d) + '.pth', map_location=device))

    """ adding DP noise to sensitive data """
    noise_mean_emb1_1 = mean_emb1_1
    noise_mean_emb1_2 = mean_emb1_2
    if ar.is_private and ar.tgt_eps is not None:
        privacy_param = privacy_calibrator.gaussian_mech(ar.tgt_eps, ar.tgt_delta, k=2)
        # TODO: fix this
        print(f'eps,delta = ({ar.tgt_eps},{ar.tgt_delta}) ==> Noise level sigma=', privacy_param['sigma'])
        noise = torch.randn(mean_emb1_1.shape[0], device=device)
        #noise = torch.randn(mean_emb1_1.shape[0], mean_emb1_1.shape[1], device=device)
        std = 2 * privacy_param['sigma'] / n_data
        noise = noise * std
        noise_mean_emb1_1 += noise

    """ adding DP noise to sensitive data """
    if ar.data == 'cifar10':
        code_fun = get_code_fun(device, n_classes, ar.d_code)
    elif ar.data == 'celeba':
        code_fun = get_code_fun(device, n_classes, ar.d_code)
    else:
        code_fun = model.get_code

    """ generate 100 sample for output log """
    fixed_labels = pt.repeat_interleave(pt.arange(1, device=device), ar.batch_size)
    print(fixed_labels.shape)
    fixed_noise = code_fun(batch_size=ar.batch_size, labels=fixed_labels[:, None])

    """ training a Generator via minimizing MMD """
    for epoch in range(ar.n_iter):  # loop over the dataset multiple times
        running_loss = 0.0
        optimizer.zero_grad()  # zero the parameter gradients

        """ synthetic data """
        gen_code, _ = code_fun(ar.batch_size)
        gen_code = gen_code.to(device)
        gen_samples = model(gen_code)

        """ synthetic data mean_emb init """
        mean_emb2_1 = torch.zeros(d, device=device)
        mean_emb2_2 = torch.zeros((32 * 32, 3), device=device)
        for idx in range(gen_samples.shape[0]):
            """ manually set the weight if needed """
            # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[gen_labels_numerical[idx], :][None, :])
            mean_v_samp = torch.Tensor([]).to(device)  # sample mean vector init
            if ar.data == 'cifar10':
                f_x = model_ntk(gen_samples[idx][None, :, :, :])  # 1 input, dimensions need tweaking
            # elif ar.data == 'celeba':
            #     f_x = model_ntk(gen_samples[idx][None, :, :, :])
            else:
                f_x = model_ntk(gen_samples[idx][None, :])

            """ get NTK features """
            f_idx_grad = torch.autograd.grad(f_x, model_ntk.parameters(),
                                             grad_outputs=f_x.data.new(f_x.shape).fill_(1), create_graph=True)
            for g in f_idx_grad:
                mean_v_samp = torch.cat((mean_v_samp, g.flatten()))
            mean_v_samp = mean_v_samp[:-1]

            """ normalize the sample mean vector """
            if ar.is_private:
                #mean_emb2_1 += mean_v_samp
                mean_emb2_1 += mean_v_samp / (torch.norm(mean_v_samp)*ar.norm_param)
            else:
                mean_emb2_1 += mean_v_samp
            mean_emb2_2 += torch.transpose(torch.flatten(gen_samples[idx], 1), 0, 1)

        """ average by batch size """
        mean_emb2_1 = torch.div(mean_emb2_1, ar.batch_size)
        mean_emb2_2 = torch.div(mean_emb2_2, ar.batch_size)

        loss_1 = torch.norm(noise_mean_emb1_1 - mean_emb2_1, p=2) ** 2
        # loss_2 = torch.sum(pt.matmul(torch.abs(noise_mean_emb1_2 - mean_emb2_2), reg))
        reg_1 = torch.norm((noise_mean_emb1_2[:, 0] - mean_emb2_2[:, 0]), p=2) ** 2
        reg_2 = torch.norm((noise_mean_emb1_2[:, 1] - mean_emb2_2[:, 1]), p=2) ** 2
        reg_3 = torch.norm((noise_mean_emb1_2[:, 2] - mean_emb2_2[:, 2]), p=2) ** 2
        # mean_emb1_vec = (noise_mean_emb1_2[:, 0] + noise_mean_emb1_2[:, 1] + noise_mean_emb1_2[:, 2]) / 3
        # mean_emb2_vec = (mean_emb2_2[:, 0] + mean_emb2_2[:, 1] + mean_emb2_2[:, 2]) / 3
        # loss_2 = torch.norm((mean_emb1_vec - mean_emb2_vec), p=2) ** 2 * reg[0]
        """ calculate loss """
        # loss_var_l1, loss_var_l2 = get_image_prior_losses(gen_samples)
        # loss = (loss_1 + reg_1 + reg_2 + reg_3) *ar.reg
        # loss = torch.sum(torch.abs(noise_mean_emb1 - mean_emb2))
        # loss = loss_1 + loss_2
        loss = loss_1
        # loss = loss_1 + ar.reg * loss_var_l1 + ar.reg * loss_var_l2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if running_loss <= 1e-4:
            break
        if (epoch) % ar.log_interval == 0 or epoch==ar.n_iter-1:
            print(loss_1)
            # print(loss_var_l1)
            # print(loss_var_l2)
            # print(reg_1)
            # print(reg_2)
            # print(reg_3)
            # print(loss_2)
            if ar.data == 'cifar10' or ar.data == 'celeba':
                log_step(epoch, loss, model, fixed_noise, 'test', ar)
            else:
                log_gen_data(model, device, epoch, n_classes, ar.log_dir)
        if epoch % ar.scheduler_interval == 0:
            scheduler.step()
        # print('epoch # and running loss are ', [epoch, running_loss])
        # training_loss_per_epoch[epoch] = running_loss

    """ log outputs """
    if ar.data == 'cifar10' or ar.data == 'celeba':
        # save trained model and data
        os.makedirs(ar.log_dir + "generated/", exist_ok=True)
        if ar.id == "None":
            ar.id=""
        pt.save(model.state_dict(), ar.log_dir + "generated/"+ ar.id+ 'gen.pt')
        data_file = os.path.join(ar.log_dir+"generated/", ar.id+'gen_data.npz')
        syn_data_size = 5000
        syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, code_fun,
                                                                   ar.gen_batch_size, syn_data_size,
                                                                   n_classes)

        np.savez(data_file, x=syn_data, y=syn_labels)

        fid_score = get_fid_scores(data_file, ar.data, device, syn_data_size,
                                   image_size, center_crop_size=32, use_autoencoder=False,
                                   base_data_dir='../data', batch_size=50)
        print(f'fid={fid_score}')
        np.save(os.path.join(ar.log_dir, 'fid.npy'), fid_score)
    else:
        """plot generation """
        log_gen_data(model, device, ar.n_iter, n_classes, ar.log_dir)

        """evaluate the model"""
        pt.save(model.state_dict(), ar.log_dir + 'gen.pt')
        syn_data, syn_labels = synthesize_mnist_with_uniform_labels(model, device)
        np.savez(ar.log_dir + 'synthetic_mnist', data=syn_data, labels=syn_labels)
        final_score = test_gen_data(ar.log_dir, ar.data, data_base_dir="", subsample=0.1, custom_keys="logistic_reg",
                                    data_from_torch=True)
        log_final_score(ar.log_dir, final_score)

        final_score = test_gen_data(ar.log_dir, ar.data, data_base_dir="", subsample=0.1, custom_keys="mlp",
                                    data_from_torch=True)
        log_final_score(ar.log_dir, final_score)

    return fid_score, 0
