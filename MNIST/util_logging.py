import os
import sys
import colorlog
import torch as pt
import numpy as np
from torchvision import utils as vutils
from torchvision import datasets
from synth_data_benchmark import prep_models, model_test_run

LOG = colorlog.getLogger(__name__)


def get_base_log_dir():
    # logdir_candidates = ['anon']
    default_base_dir = os.path.normpath(os.path.join(os.getcwd(), '../logs'))
    # for path in logdir_candidates:
    #   if os.path.exists(path):
    #     return path
    if os.path.exists(default_base_dir):
        return default_base_dir
    else:
        return None


def configure_logger(log_importance_level):
    levels = {'debug': colorlog.DEBUG,
              'info': colorlog.INFO,
              'warning': colorlog.WARNING}
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s',
                                                   log_colors={'DEBUG': 'cyan', 'INFO': 'green',
                                                               'WARNING': 'yellow', 'ERROR': 'red',
                                                               'CRITICAL': 'bold_red'}))
    handler.setLevel(levels[log_importance_level])
    LOG.addHandler(handler)
    LOG.setLevel(levels[log_importance_level])
    LOG.info("begin logging")


def log_noise_norm(noise_vec, data_norm, writer, prefix):
    noise_norm = pt.linalg.norm(noise_vec)
    snr = data_norm / noise_norm
    if writer is not None:
        writer.add_scalar(f'{prefix}/real_feature_norm', data_norm, global_step=0)
        writer.add_scalar(f'{prefix}/noise_vec_norm', noise_norm, global_step=0)
        writer.add_scalar(f'{prefix}/signal_noise_ratio', snr, global_step=0)
        LOG.info(f'{prefix} feat norm: {data_norm}, noise norm: {noise_norm}, SNR: {snr}')


def log_losses_and_imgs(net_gen, train_acc_losses, valid_acc_losses, fixed_noise,
                        iter_id, writer, n_iter, valid_iter, log_dir, exp_name):
    ngm = train_acc_losses.gen_mean / valid_iter
    ngv = train_acc_losses.var / valid_iter
    nm = train_acc_losses.mean / valid_iter
    nv = train_acc_losses.var / valid_iter
    LOG.info(f'[{iter_id + 1}/{n_iter}] train Loss_Gz: {ngm:.6f} Loss_GzVar: {ngv:.6f} '
             f'Loss_vMean: {nm:.6f} Loss_vVar: {nv:.6f}')
    if writer is not None:
        writer.add_scalar('train_losses/generator_mean_matching', ngm, global_step=iter_id)
        writer.add_scalar('train_losses/generator_var_matching', ngv, global_step=iter_id)
        writer.add_scalar('train_losses/mean_net', nm, global_step=iter_id)
        writer.add_scalar('train_losses/var_net', nv, global_step=iter_id)

    if valid_acc_losses is not None:
        ngm = valid_acc_losses.gen_mean / valid_iter
        ngv = valid_acc_losses.var / valid_iter
        nm = valid_acc_losses.mean / valid_iter
        nv = valid_acc_losses.var / valid_iter
        LOG.info(' ' * len(f'[{iter_id + 1}/{n_iter}] ') +
                 f'valid Loss_Gz: {ngm:.6f} Loss_GzVar: {ngv:.6f} '
                 f'Loss_vMean: {nm:.6f} Loss_vVar: {nv:.6f}')
        if writer is not None:
            writer.add_scalar('validation_losses/generator_mean_matching', ngm, global_step=iter_id)
            writer.add_scalar('validation_losses/generator_var_matching', ngv, global_step=iter_id)
            writer.add_scalar('validation_losses/mean_net', nm, global_step=iter_id)
            writer.add_scalar('validation_losses/var_net', nv, global_step=iter_id)

    with pt.no_grad():
        if isinstance(net_gen, pt.nn.parallel.DistributedDataParallel):
            fake = net_gen.module(fixed_noise).detach()
        else:
            fake = net_gen(fixed_noise).detach()

    if exp_name is None:
        img_path = f'{log_dir}/images/fake_samples_iterId_{iter_id + 1}.png'
    else:
        img_path = f"{log_dir}/images/{exp_name.replace('/', '_', 999)}_it_{iter_id + 1}.png"

    vutils.save_image(fake.data[:100], img_path, normalize=True, nrow=10)

    sys.stdout.flush()
    if writer is not None:
        writer.flush()


def log_real_feature_norms(l2_norms, l2_norms_sqrd, writer):
    if writer is None:
        return

    writer.add_histogram('real_feature_norms', l2_norms, global_step=0)
    writer.add_scalar('real_feature_norm/min', pt.min(l2_norms), global_step=0)
    writer.add_scalar('real_feature_norm/mean', pt.mean(l2_norms), global_step=0)
    writer.add_scalar('real_feature_norm/max', pt.max(l2_norms), global_step=0)

    if isinstance(l2_norms_sqrd, pt.Tensor):
        writer.add_histogram('real_feature_norms_sqrd', l2_norms_sqrd, global_step=0)
        writer.add_scalar('real_feature_norm_sqrd/min', pt.min(l2_norms_sqrd), global_step=0)
        writer.add_scalar('real_feature_norm_sqrd/mean', pt.mean(l2_norms_sqrd), global_step=0)
        writer.add_scalar('real_feature_norm_sqrd/max', pt.max(l2_norms_sqrd), global_step=0)


def log_fake_feature_norms(l2_norms, l2_norms_sqrd, iter_id, writer):
    if writer is None:
        return

    if isinstance(l2_norms, pt.Tensor):
        writer.add_histogram('fake_feature_norms', l2_norms, global_step=iter_id)
        writer.add_scalar('fake_feature_norm/min', pt.min(l2_norms), global_step=iter_id)
        writer.add_scalar('fake_feature_norm/mean', pt.mean(l2_norms), global_step=iter_id)
        writer.add_scalar('fake_feature_norm/max', pt.max(l2_norms), global_step=iter_id)

    if isinstance(l2_norms_sqrd, pt.Tensor):
        writer.add_histogram('fake_feature_norms_sqrd', l2_norms_sqrd, global_step=iter_id)
        writer.add_scalar('fake_feature_norm_sqrd/min', pt.min(l2_norms_sqrd), global_step=iter_id)
        writer.add_scalar('fake_feature_norm_sqrd/mean', pt.mean(l2_norms_sqrd), global_step=iter_id)
        writer.add_scalar('fake_feature_norm_sqrd/max', pt.max(l2_norms_sqrd), global_step=iter_id)


def route_io_to_file(logdir, out_file_name, err_file_name):
    sys.stdout = open(os.path.join(logdir, out_file_name), 'a')
    sys.stderr = open(os.path.join(logdir, err_file_name), 'a')


def delayed_log(level, message):
    log_actions = {'debug': LOG.debug, 'info': LOG.info, 'warning': LOG.warning, 'error': LOG.error}
    assert level in log_actions
    log_actions[level](message)


def create_synth_dataset(n_samples, net_gen, batch_size, noise_maker, device, data_format='array',
                         save_dir='.', file_name='synth_data', n_classes=None):
    assert data_format in {'array', 'tensor'}, f'wrong format: {data_format}'
    batches = [batch_size] * (n_samples // batch_size)
    if n_samples % batch_size > 0:
        batches.append(n_samples % batch_size)
    if n_classes is not None:
        balanced_labels_int = pt.repeat_interleave(pt.arange(0, n_classes, dtype=pt.int64,
                                                             device=device),
                                                   n_samples // n_classes)
        random_labels_int = pt.randint(n_classes, (n_samples % n_classes,),
                                       dtype=pt.int64, device=device)
        labels_int = pt.cat([balanced_labels_int, random_labels_int])
        labels_list = pt.split(pt.eye(n_classes, device=device)[labels_int], batches)
    else:
        labels_int = None
        labels_list = [None] * len(batches)
    samples_list = []
    with pt.no_grad():
        for labels in labels_list:
            z_in = noise_maker.noise_fun(labels=labels)
            if isinstance(net_gen, pt.nn.parallel.DistributedDataParallel):
                syn_batch = net_gen.module(z_in)  # don't sync, since this only runs on one gpu
            else:
                syn_batch = net_gen(z_in)

            samples_list.append(syn_batch.detach().cpu())
        samples = pt.cat(samples_list, dim=0)
    if data_format == 'array':
        file_name = file_name if file_name.endswith('.npz') else file_name + '.npz'
        file_path = os.path.join(save_dir, file_name)
        if labels_int is None:
            np.savez(file_path, x=samples.numpy())
        else:
            np.savez(file_path, x=samples.numpy(), y=labels_int.cpu().numpy())
    else:
        file_name = file_name if file_name.endswith('.pt') else file_name + '.pt'
        file_path = os.path.join(save_dir, file_name)
        if labels_int is None:
            pt.save({'x': samples}, file_path)
        else:
            pt.save({'x': samples, 'y': labels_int.cpu()}, file_path)
    return file_path


def mnist_synth_to_real_test(dataset, data_path, writer, log_dir, acc_file_name, step):
    # load test data
    if dataset == 'dmnist':
        test_data = datasets.MNIST('../data', train=False, download=True)
    elif dataset == 'fmnist':
        test_data = datasets.FashionMNIST('../data', train=False, download=True)
    else:
        raise ValueError
    x_real_test, y_real_test = test_data.data.numpy(), test_data.targets.numpy()
    x_real_test = np.reshape(x_real_test, (-1, 784)) / 255

    # load synthetic data
    syn_data_dict = np.load(data_path)
    x_syn = syn_data_dict['x']
    y_syn = syn_data_dict['y'] if 'y' in syn_data_dict else None
    x_syn = np.reshape(x_syn, (-1, 784))
    print(f'syn  range: {np.max(x_syn)}, {np.mean(x_syn)}, {np.min(x_syn)}')
    print(f'real range: {np.max(x_real_test)}, {np.mean(x_real_test)}, {np.min(x_real_test)}')
    print(f'y shapes: real {y_real_test.shape}, syn {y_syn.shape}')
    if len(y_syn.shape) == 2:  # remove onehot
        if y_syn.shape[1] == 1:
            y_syn = y_syn.ravel()
        elif y_syn.shape[1] == 10:
            y_syn = np.argmax(y_syn, axis=1)
        else:
            raise ValueError
    classifier_names = ['logistic_reg', 'mlp', 'xgboost']
    mean_acc, accs = test_passed_gen_data(x_syn, y_syn, x_real_test, y_real_test,
                                          custom_keys=','.join(classifier_names))
    accs_by_classifier = {cn: acc for cn, acc in zip(classifier_names, accs)}
    np.savez(os.path.join(log_dir, acc_file_name),
             mean_acc=mean_acc, **accs_by_classifier)
    if writer is not None:
        writer.add_scalar('eval/mean_acc', mean_acc, global_step=step)
        for k, v in accs_by_classifier.items():
            writer.add_scalar(f'eval/{k}_acc', v, global_step=step)
    accs_str = ','.join([f'{k}={v}' for k, v in accs_by_classifier.items()])
    LOG.info(f'mean accuracy: {mean_acc}, individual accuracies: {accs_str}')
    return mean_acc, accs


def test_passed_gen_data(x_syn, y_syn, x_real_test, y_real_test,
                         custom_keys=None,
                         norm_data=False):
    models, model_specs, run_keys = prep_models(custom_keys, False, False, False)

    g_to_r_acc_summary = []
    for key in run_keys:
        model = models[key](**model_specs[key])
        g_to_r_acc, _, _, _, _ = model_test_run(model, x_syn, y_syn, x_real_test, y_real_test,
                                                norm_data, '', '')
        g_to_r_acc_summary.append(g_to_r_acc)

    mean_acc = np.mean(g_to_r_acc_summary)
    return mean_acc, g_to_r_acc_summary
