import os.path

import torch as pt
import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np


def get_mnist_dataloaders(batch_size, test_batch_size, use_cuda, normalize=False,
                          dataset='dmnist', data_dir='data', flip=False, return_datasets=False):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transforms_list = [transforms.ToTensor()]
    if dataset == 'dmnist':
        if normalize:
            mnist_mean = 0.1307
            mnist_sdev = 0.3081
            transforms_list.append(transforms.Normalize((mnist_mean,), (mnist_sdev,)))
        prep_transforms = transforms.Compose(transforms_list)
        trn_data = dset.MNIST(data_dir, train=True, download=True, transform=prep_transforms)
        tst_data = dset.MNIST(data_dir, train=False, transform=prep_transforms)
        if flip:
            assert not normalize
            print(pt.max(trn_data.data))
            flip_mnist_data(trn_data)
            flip_mnist_data(tst_data)

        train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
    elif dataset == 'fmnist':
        assert not normalize
        prep_transforms = transforms.Compose(transforms_list)
        trn_data = dset.FashionMNIST(data_dir, train=True, download=True, transform=prep_transforms)
        tst_data = dset.FashionMNIST(data_dir, train=False, transform=prep_transforms)
        if flip:
            print(pt.max(trn_data.data))
            flip_mnist_data(trn_data)
            flip_mnist_data(tst_data)
        train_loader = pt.utils.data.DataLoader(trn_data, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = pt.utils.data.DataLoader(tst_data, batch_size=test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError

    if return_datasets:
        return train_loader, test_loader, trn_data, tst_data
    else:
        return train_loader, test_loader


def flip_mnist_data(dataset):
    data = dataset.data
    flipped_data = 255 - data
    selections = np.zeros(data.shape[0], dtype=np.int)
    selections[:data.shape[0] // 2] = 1
    selections = pt.tensor(np.random.permutation(selections), dtype=pt.uint8)
    print(selections.shape, data.shape, flipped_data.shape)
    dataset.data = pt.where(selections[:, None, None], data, flipped_data)

