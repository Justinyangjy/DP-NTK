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
from models.generators import ResnetG, ConvCondGen, Generative_Model_homogeneous_data, Generative_Model_heterogeneous_data
from models.ntk import *
from synth_data_benchmark import test_gen_data
from util import plot_mnist_batch, log_final_score
from all_aux_tab import test_models

heterogeneous_datasets = ["adult", "census", "cervical", "intrusion", "covtype"]
homogeneous_datasets = ["credit", "isolet", "epileptic"]

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

def get_code_tab_fun(device, batch_size, d_code, labels_distribution):
    label_input = torch.multinomial(torch.Tensor(labels_distribution), batch_size, replacement=True).type(torch.FloatTensor)

    label_input = torch.cat([label_input, torch.arange(len(labels_distribution), out=torch.FloatTensor())])  # to avoid no labels
    #label_input = label_input.transpose_(0, 1)
    #label_input = label_input.squeeze()
    label_input = label_input.to(device)

    # (2) generate corresponding features
    # batch_size + len(weights) # batch required + all labels to avoid no labels
    feature_input = torch.randn((batch_size + len(labels_distribution), d_code - 1)).to(device)
    input_to_model = torch.cat((feature_input, label_input[:, None]), 1)

    return input_to_model, label_input


def log_step(step, loss, gen, fixed_noise, exp_name, log_dir):
    print(f'Train step: {step} \tLoss: {loss.item():.6f}')
    with pt.no_grad():
        fake = gen(fixed_noise).detach()
    img_dir = os.path.join(log_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"{exp_name.replace('/', '_', 999)}_it_{step + 1}.png")

    vutils.save_image(fake.data[:100], img_path, normalize=True, nrow=10)


def synthesize_data_with_uniform_labels(gen, device, code_fun, gen_batch_size=1000, n_data=60000, n_labels=10):
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

def synthesize_data_tab(ar, model, data_name, n, input_size, num_numerical_inputs, labels_distribution, n_classes, device):



    """ Once the training step is over, we produce 60K samples and test on downstream tasks """
    """ now we save synthetic data of size 60K and test them on logistic regression """
    #######################################################################33
    if data_name in heterogeneous_datasets:

        """ draw final data samples """

        label_input = torch.multinomial(torch.Tensor([labels_distribution]), n, replacement=True).type(torch.FloatTensor)
        label_input = label_input.transpose_(0, 1)
        label_input = label_input.to(device)

        # (2) generate corresponding features
        feature_input = torch.randn((n, ar.d_code - 1)).to(device)
        input_to_model = torch.cat((feature_input, label_input), 1)

        # input_to_model, labels = get_code_tab_fun(device, ar.batch_size, ar.d_code, labels_distribution)

        outputs = model(input_to_model)

        samp_input_features = outputs
        samp_labels = label_input

        # (3) round the categorial features
        output_numerical = outputs[:, 0:num_numerical_inputs]
        output_categorical = outputs[:, num_numerical_inputs:]
        output_categorical = torch.round(output_categorical)

        output_combined = torch.cat((output_numerical, output_categorical), 1)

        generated_input_features_final = output_combined.cpu().detach().numpy()
        generated_labels_final = label_input.cpu().detach().numpy()

        return generated_input_features_final, generated_labels_final




    else:  # homogeneous datasets

        """ draw final data samples """
        label_input = (1 * (torch.rand((n)) < labels_distribution[1])).type(torch.FloatTensor)
        # label_input = torch.multinomial(torch.Tensor([weights]), n, replacement=True).type(torch.FloatTensor)
        # label_input = torch.multinomial(1 / n_classes * torch.ones(n_classes), n, replacement=True).type(torch.FloatTensor)
        label_input = label_input.to(device)

        feature_input = torch.randn((n, input_size - 1)).to(device)
        input_to_model = torch.cat((feature_input, label_input[:, None]), 1)
        outputs = model(input_to_model)

        samp_input_features = outputs

        label_input_t = torch.zeros((n, n_classes))
        idx_1 = (label_input == 1.).nonzero()[:, 0]
        idx_0 = (label_input == 0.).nonzero()[:, 0]
        label_input_t[idx_1, 1] = 1.
        label_input_t[idx_0, 0] = 1.

        samp_labels = label_input_t

        generated_input_features_final = samp_input_features.cpu().detach().numpy()
        generated_labels_final = samp_labels.cpu().detach().numpy()
        generated_labels = np.argmax(generated_labels_final, axis=1)

        return generated_input_features_final, generated_labels


def gen_step(model_ntk, ar, device, n_train, labels_distribution=None, test_data=None, test_labels=None):
    random.seed(ar.seed)
    pt.manual_seed(ar.seed)

    """ setup """
    tab_datasets = ["adult", "census", "cervical", "credit", "isolet", "epileptic", "intrusion", "covtype"]
    if ar.data == 'cifar10':
        input_dim = 32 * 32 * 3
        image_size = 32
        n_data = 50_000
        n_classes = 10
    elif "mnist" in ar.data:
        input_dim = 784
        image_size = 28
        n_data = 60_000
        n_classes = 10
    elif ar.data in tab_datasets:
        n_data = n_train
        input_dim = test_data.shape[1]
        n_classes = len(set(test_labels))

    # generative models
    print('data: ', ar.data)
    if ar.data == 'cifar10':
        model = ResnetG(ar.d_code + n_classes, nc=3, ndf=64, image_size=32,
                        adapt_filter_size=True,
                        use_conv_at_skip_conn=False).to(device)
    elif "mnist" in ar.data:
        model = ConvCondGen(ar.d_code, ar.gen_spec, n_classes, '16,8', '5,5').to(device)
    else:
        #batch_size = np.int(np.round(batch_rate * n))
        input_size = ar.d_code
        hidden_size_1 = 4 * input_dim
        hidden_size_2 = 2 * input_dim

        if ar.data in heterogeneous_datasets:
            categories_num={"adult": [8,6], "census": [33,7], "cervical": [23,11], "covtype": [44,10], "intrusion": [36,4]} # cat, num
            num_categorical_inputs = categories_num[ar.data][0]
            num_numerical_inputs = categories_num[ar.data][1]
            output_size = num_categorical_inputs + num_numerical_inputs
            model = Generative_Model_heterogeneous_data(input_size=input_size,hidden_size_1=hidden_size_1,hidden_size_2=hidden_size_2,output_size=output_size,num_categorical_inputs=num_categorical_inputs,num_numerical_inputs=num_numerical_inputs).to(device)
        elif ar.data in homogeneous_datasets:
            num_numerical_inputs = input_dim
            model = Generative_Model_homogeneous_data(input_size=input_size, hidden_size_1=hidden_size_1,hidden_size_2=hidden_size_2, output_size=input_dim, dataset=ar.data).to(device) #input_dim of data is output for generated data




    optimizer = optim.Adam(model.parameters(), lr=ar.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
    training_loss_per_epoch = np.zeros(ar.n_iter)

    """ initialize the variables """
    mean_v_samp = torch.Tensor([]).to(device)
    for p in model_ntk.parameters():
        mean_v_samp = torch.cat((mean_v_samp, p.flatten()))
    d = len(mean_v_samp) - 1
    print('Feature Length:', d)
    mean_emb1 = pt.load(ar.log_dir + 'mean_emb1_' + str(d) + '.pth')
    model_ntk.load_state_dict(pt.load(ar.log_dir + 'model_' + str(d) + '.pth', map_location=device))

    ####################################################
    # Privatising quantities

    if "mnist" in ar.data or "cifar" in ar.data: #balanced datasets
        """ adding DP noise to sensitive data """
        noise_mean_emb1 = mean_emb1
        if ar.tgt_eps is not None:
            privacy_param = privacy_calibrator.gaussian_mech(ar.tgt_eps, ar.tgt_delta, k=1)
            print(f'eps,delta = ({ar.tgt_eps},{ar.tgt_delta}) ==> Noise level sigma=', privacy_param['sigma'])
            noise = torch.randn(mean_emb1.shape[0], mean_emb1.shape[1], device=device)
            std = 2 * privacy_param['sigma'] / n_data
            noise = noise * std
            noise_mean_emb1 += noise


    else: #tabular datasets which are not balanced,
        # privatizing weights (label distribution) and mean embeddings
        """ specifying ratios of data to generate depending on the class labels """
        unnormalized_weights = labels_distribution
        weights = unnormalized_weights / np.sum(unnormalized_weights)
        print('\nweights with no privatization are', weights, '\n')



        """ privatizing weights """

        print("private")
        # desired privacy level
        epsilon = ar.tgt_eps
        delta = 1e-5
        k = n_classes   # +1? this dp analysis has been updated
        #k = 2
        privacy_param = privacy_calibrator.gaussian_mech(epsilon, delta, k=k)
        print(f'eps,delta = ({epsilon},{delta}) ==> Noise level sigma=', privacy_param['sigma'])

        sensitivity_for_weights = np.sqrt(2) / n_data  # double check if this is sqrt(2) or 2
        noise_std_for_weights = privacy_param['sigma'] * sensitivity_for_weights
        weights = weights + np.random.randn(weights.shape[0]) * noise_std_for_weights
        weights[weights < 0] = 1e-3  # post-processing so that we don't have negative weights.
        print('weights after privatization are', weights)

        """ privatizing each column of mean embedding """

        if ar.data in heterogeneous_datasets:
            sensitivity = 2 * np.sqrt(2) / n_data
        else:
            sensitivity = 2 / n_data
        noise_std_for_privacy = privacy_param['sigma'] * sensitivity

        # make sure add noise after rescaling
        weights_torch = torch.Tensor(weights)
        weights_torch = weights_torch.to(device)

        rescaled_mean_emb = weights_torch * mean_emb1
        noise = noise_std_for_privacy * torch.randn(mean_emb1.size())
        noise = noise.to(device)

        rescaled_mean_emb = rescaled_mean_emb + noise

        noise_mean_emb1 = rescaled_mean_emb / weights_torch  # rescaling back\

    # End of Privatising quantities if necessary
    ####################################################
###############



    if ar.data == 'cifar10':
        code_fun = get_code_fun(device, n_classes, ar.d_code)
    elif ar.data in tab_datasets:
        code_fun, labels = get_code_tab_fun(device, ar.gen_batch_size, ar.d_code, labels_distribution) #fig out arguments
    else:
        code_fun = model.get_code

    """ generate 100 sample for output log """
    if ar.data == "cifar10" or "mnist"in ar.data:
        fixed_labels = pt.repeat_interleave(pt.arange(10, device=device), 10)
        print(fixed_labels.shape)
        fixed_noise = code_fun(batch_size=100, labels=fixed_labels[:, None])


    """ training a Generator via minimizing MMD """
    print("\n\nTraining generator\n")
    for epoch in range(ar.n_iter):  # loop over the dataset multiple times
        running_loss = 0.0
        optimizer.zero_grad()  # zero the parameter gradients

        """ synthetic data """
        if ar.data == "cifar10" or "mnist" in ar.data:
            gen_code, gen_labels = code_fun(ar.batch_size)
            gen_code = gen_code.to(device)
            gen_samples = model(gen_code)
            _, gen_labels_numerical = torch.max(gen_labels, dim=1)
        else: #tab
            gen_code = code_fun
            #print("gen input", gen_code.shape)
            gen_samples = model(gen_code)
            #print("gen output", gen_samples.shape)

        """ synthetic data mean_emb init """
        mean_emb2 = torch.zeros((d, n_classes), device=device)
        for idx in range(gen_samples.shape[0]):
            """ manually set the weight if needed """
            # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[gen_labels_numerical[idx], :][None, :])
            mean_v_samp = torch.Tensor([]).to(device)  # sample mean vector init
            if ar.data == 'cifar10':
                f_x = model_ntk(gen_samples[idx][None, :, :, :])  # 1 input, dimensions need tweaking
            elif ar.data in tab_datasets:
                f_x = model_ntk(gen_samples[idx][None, :])
            else:
                f_x = model_ntk(gen_samples[idx][None, :])

            """ get NTK features """
            f_idx_grad = torch.autograd.grad(f_x, model_ntk.parameters(),
                                             grad_outputs=f_x.data.new(f_x.shape).fill_(1), create_graph=True)
            for g in f_idx_grad:
                mean_v_samp = torch.cat((mean_v_samp, g.flatten()))
            mean_v_samp = mean_v_samp[:-1]

            """ normalize the sample mean vector """
            m = mean_v_samp
            #m = mean_v_samp / torch.norm(mean_v_samp)
            if ar.data in tab_datasets:
                mean_emb2[:, labels[idx].long()] += m
            else:
                mean_emb2[:, gen_labels_numerical[idx].long()] += m

        """ average by batch size """
        mean_emb2 = torch.div(mean_emb2, ar.gen_batch_size)

        """ calculate loss """
        loss = torch.norm(noise_mean_emb1 - mean_emb2, p=2) ** 2
        # loss = torch.sum(torch.abs(noise_mean_emb1 - mean_emb2))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if running_loss <= 1e-4:
            break
        if epoch % 100 == 0:
            if ar.data == 'cifar10':
                log_step(epoch, loss, model, fixed_noise, 'test', ar.log_dir)
            # else:
            #     log_gen_data(model, device, epoch, n_classes, ar.log_dir)
        if epoch % ar.scheduler_interval == 0:
            scheduler.step()
        print('epoch # and running loss are ', [epoch, running_loss])
        # training_loss_per_epoch[epoch] = running_loss



    print("\n\nTesting\n")
    """ log outputs """
    score=None
    if ar.data == 'cifar10':
        # save trained model and data
        pt.save(model.state_dict(), ar.log_dir + 'gen.pt')

        data_file = os.path.join(ar.log_dir, 'gen_data.npz')
        syn_data_size = 5000
        syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, code_fun,
                                                                   ar.gen_batch_size, syn_data_size,
                                                                   n_classes)

        np.savez(data_file, x=syn_data, y=syn_labels)

        fid_score = get_fid_scores(data_file, 'cifar10', device, syn_data_size,
                                   image_size, center_crop_size=32, use_autoencoder=False,
                                   base_data_dir='../data', batch_size=50)
        print(f'fid={fid_score}')
        np.save(os.path.join(ar.log_dir, 'fid.npy'), fid_score)
    elif "mnist" in ar.data:
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
        
    elif ar.data in tab_datasets:

        generated_input_features_final, generated_labels = synthesize_data_tab(ar, model, ar.data, n_train, input_size, num_numerical_inputs, labels_distribution, n_classes, device)

        roc, prc = test_models(generated_input_features_final, generated_labels, test_data, test_labels, n_classes, "generated", ar.tab_classifiers, ar.data)

        score = roc,prc

    return score
