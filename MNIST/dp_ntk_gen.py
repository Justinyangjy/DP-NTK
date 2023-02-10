import random

import numpy as np
import torch as pt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models.generators import ConvCondGen
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


def gen_step(model_ntk, ar, device):
    random.seed(ar.seed)
    pt.manual_seed(ar.seed)

    """ setup """
    input_dim = 784
    image_size = 28
    n_data = 60_000
    n_classes = 10

    print('data: ', ar.data)
    model = ConvCondGen(ar.d_code, ar.gen_spec, n_classes, '16,8', '5,5').to(device)

    optimizer = optim.Adam(model.parameters(), lr=ar.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
    training_loss_per_epoch = np.zeros(ar.n_iter)

    """ initialize the variables """
    mean_v_samp = pt.Tensor([]).to(device)
    for p in model_ntk.parameters():
        mean_v_samp = pt.cat((mean_v_samp, p.flatten()))
    d = len(mean_v_samp)
    print('Feature Length:', d)

    if ar.tgt_eps is None:
        noise_mean_emb1 = pt.load(ar.save_dir + f'mean_emb1_{d}_{ar.seed}.pth')
    else:
        noise_mean_emb1 = pt.load(ar.save_dir + f'mean_emb1_{d}_{ar.seed}_{ar.tgt_eps}.pth')
    model_ntk.load_state_dict(pt.load(ar.save_dir + f'model_{d}_{ar.seed}.pth', map_location=device))

    """ random noise func for generators """
    code_fun = model.get_code

    """ generate 100 samples for output log """
    fixed_labels = pt.repeat_interleave(pt.arange(10, device=device), 10)
    print(fixed_labels.shape)
    fixed_noise = code_fun(batch_size=100, labels=fixed_labels[:, None])

    """ training a Generator via minimizing MMD """
    for epoch in range(ar.n_iter):  # loop over the dataset multiple times
        running_loss = 0.0
        optimizer.zero_grad()  # zero the parameter gradients

        """ synthetic data """
        gen_code, gen_labels = code_fun(ar.batch_size)
        gen_code = gen_code.to(device)
        gen_samples = model(gen_code)
        _, gen_labels_numerical = pt.max(gen_labels, dim=1)

        """ synthetic data mean_emb init """
        mean_emb2 = pt.zeros((d, n_classes), device=device)
        for idx in range(gen_samples.shape[0]):
            """ manually set the weight if needed """
            # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[gen_labels_numerical[idx], :][None, :])
            mean_v_samp = pt.Tensor([]).to(device)  # sample mean vector init
            f_x = model_ntk(gen_samples[idx][None, :])

            """ get NTK features """
            f_idx_grad = pt.autograd.grad(f_x, model_ntk.parameters(),
                                             grad_outputs=f_x.data.new(f_x.shape).fill_(1), create_graph=True)
            for g in f_idx_grad:
                mean_v_samp = pt.cat((mean_v_samp, g.flatten()))
            # mean_v_samp = mean_v_samp[:-1]

            """ normalize the sample mean vector """
            mean_emb2[:, gen_labels_numerical[idx]] += mean_v_samp / pt.linalg.vector_norm(mean_v_samp)

        """ average by batch size """
        mean_emb2 = pt.div(mean_emb2, ar.batch_size)

        """ calculate loss """
        loss = pt.norm(noise_mean_emb1 - mean_emb2, p=2) ** 2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (epoch + 1) % ar.log_interval == 0:
            log_gen_data(model, device, epoch, n_classes, ar.log_dir)
            print('epoch # and running loss are ', [epoch, running_loss])
            training_loss_per_epoch[epoch] = running_loss
        if epoch % ar.scheduler_interval == 0:
            scheduler.step()

    """ log outputs """
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
