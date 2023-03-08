import os
import numpy as np
import torch as pt
from torchvision import transforms
import torchvision.datasets as dset
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm
from data_loading import load_synth_dataset, load_dataset


def cifar10_stats(model, device, batch_size, workers, image_size=32, dataroot='../data'):
    transformations = [transforms.Resize(image_size), transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]
    dataset = dset.CIFAR10(root=dataroot, download=True,
                           transform=transforms.Compose(transformations))
    assert dataset
    # noinspection PyUnresolvedReferences
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=int(workers))
    return stats_from_dataloader(dataloader, model, device)


def store_data_stats(dataset_name, image_size, center_crop_size, dataroot, use_autoencoder):
    log_dir = './data'
    device = pt.device("cuda")
    batch_size = 50
    workers = 1
    real_data_stats_dir = os.path.join(log_dir, 'fid_stats')
    os.makedirs(real_data_stats_dir, exist_ok=True)
    real_data_stats_file = os.path.join(real_data_stats_dir, dataset_name + '.npz')
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    # mu_real, sig_real = cifar10_stats(model, device, batch_size, workers,
    #                                   image_size=32, dataroot='../data')
    dataloader, _ = load_dataset(dataset_name, image_size, center_crop_size, dataroot,
                                 use_autoencoder, batch_size, workers, labeled=False,
                                 test_set=False)
    mu_real, sig_real = stats_from_dataloader(dataloader, model, device)
    np.savez(real_data_stats_file, mu=mu_real, sig=sig_real)


def get_fid_scores(synth_data_file, dataset_name, device, n_samples,
                   image_size, center_crop_size, use_autoencoder,
                   base_data_dir='./data', batch_size=50):
    real_data_stats_dir = os.path.join(base_data_dir, 'fid_stats')
    real_data_stats_file = os.path.join(real_data_stats_dir, dataset_name + '.npz')
    dims = 2048

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    if not os.path.exists(real_data_stats_file):
        store_data_stats(dataset_name, image_size, center_crop_size, base_data_dir, use_autoencoder)
        # store_data_stats(dataset_name)

    stats = np.load(real_data_stats_file)
    mu_real, sig_real = stats['mu'], stats['sig']

    # if synth_data_file.endswith('.npz'):  # allow for labels
    #   data_dict = np.load(synth_data_file)
    #   targets = data_dict['y'] if 'y' in data_dict.keys() else None
    #   synth_data = SynthDataset(data=data_dict['x'], targets=targets, to_tensor=False)
    # else:  # old version
    #   synth_data = SynthDataset(np.load(synth_data_file), targets=None, to_tensor=False)
    #
    # synth_dataloader = pt.utils.data.DataLoader(synth_data, batch_size=batch_size, shuffle=False,
    #                                                drop_last=False, num_workers=1)
    synth_data_loader = load_synth_dataset(synth_data_file, batch_size, n_samples)
    mu_syn, sig_syn = stats_from_dataloader(synth_data_loader, model, device)

    fid = calculate_frechet_distance(mu_real, sig_real, mu_syn, sig_syn)
    return fid


def stats_from_dataloader(dataloader, model, device='cpu'):
    """
  Returns:
  -- mu    : The mean over samples of the activations of the pool_3 layer of
             the inception model.
  -- sigma : The covariance matrix of the activations of the pool_3 layer of
             the inception model.
  """
    model.eval()

    pred_list = []

    start_idx = 0

    for batch in tqdm(dataloader):
        x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
        x = x.to(device)

        with pt.no_grad():
            pred = model(x)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        # pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        pred_list.append(pred)

        start_idx = start_idx + pred.shape[0]

    pred_arr = np.concatenate(pred_list, axis=0)
    # return pred_arr
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


if __name__ == '__main__':
    # store_data_stats('cifar10')
    store_data_stats('cifar10', 32, 32, '../data', False)
