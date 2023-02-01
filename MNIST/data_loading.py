import os.path

import torch as pt
import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np


def load_dataset(dataset_name, image_size, center_crop_size, dataroot, use_autoencoder, batch_size,
                 n_workers, labeled=False, test_set=False):
    if dataset_name in ['celeba']:
        n_classes = None
        transformations = []
        if center_crop_size > image_size:
            transformations.extend([transforms.CenterCrop(center_crop_size),
                                    transforms.Resize(image_size)])
        else:
            transformations.extend([transforms.Resize(image_size),
                                    transforms.CenterCrop(center_crop_size)])

        if not use_autoencoder:
            transformations.extend([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    ])
        else:
            transformations.extend([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

        # folder dataset
        dataset = dset.ImageFolder(root=os.path.join(dataroot, 'img_align_celeba'),
                                   transform=transforms.Compose(transformations))
    elif dataset_name == 'lsun':
        n_classes = None
        transformations = []
        if center_crop_size > image_size:
            transformations.extend([transforms.CenterCrop(center_crop_size),
                                    transforms.Resize(image_size)])
        else:
            transformations.extend([transforms.Resize(image_size),
                                    transforms.CenterCrop(center_crop_size)])

        if not use_autoencoder:
            transformations.extend([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    ])
        else:
            transformations.extend([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

        dataset = dset.LSUN(os.path.join(dataroot, 'lsun'), classes=['bedroom_train'],
                            transform=transforms.Compose(transformations))

    elif dataset_name == 'cifar10':
        return load_cifar10(image_size, dataroot, use_autoencoder, batch_size, n_workers, labeled,
                            test_set)
    elif dataset_name == 'stl10':
        n_classes = None
        transformations = [transforms.Resize(image_size), transforms.ToTensor()]
        if not use_autoencoder:
            transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]))
        else:
            transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        dataset = dset.STL10(root=dataroot, split='unlabeled', download=True,
                             transform=transforms.Compose(transformations))
    elif dataset_name in {'fmnist', 'dmnist', 'svhn', 'cifar10_pretrain'}:
        dataset = small_data_loader(dataset_name, not test_set)
        n_classes = 10
    elif dataset_name == 'imagenet':
        dataset = load_imagenet_subset(center_crop_size, image_size, use_autoencoder, dataroot)
        n_classes = None
    else:
        raise ValueError(f'{dataset_name} not recognized')

    assert dataset
    assert not test_set or dataset == 'cifar10'
    if labeled:
        assert n_classes is not None, 'selected dataset has no labels'
    else:
        n_classes = None

    dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=int(n_workers))

    return dataloader, n_classes


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


def load_imagenet_subset(center_crop_size, image_size, use_autoencoder, dataroot):
    transformations = []
    if center_crop_size > image_size:
        transformations.extend([transforms.CenterCrop(center_crop_size),
                                transforms.Resize(image_size)])
    else:
        transformations.extend([transforms.Resize(image_size),
                                transforms.CenterCrop(center_crop_size)])

    if not use_autoencoder:
        transformations.extend([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])
    else:
        transformations.extend([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

    # folder dataset
    dataset = dset.ImageFolder(root=os.path.join(dataroot, 'imagenet'),
                               transform=transforms.Compose(transformations))
    return dataset


def load_cifar10(image_size, dataroot, use_autoencoder, batch_size,
                 n_workers, labeled=False, test_set=False, scale_to_range=False):
    transformations = [transforms.Resize(image_size), transforms.ToTensor()]
    if scale_to_range:
        transformations.append(transforms.Lambda(lambda x: x * 2 - 1))
    else:
        transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]))

    dataset = dset.CIFAR10(root=dataroot, train=not test_set, download=True,
                           transform=transforms.Compose(transformations))

    n_classes = 10 if labeled else None
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=int(n_workers))

    return dataloader, n_classes


def load_synth_dataset(data_file, batch_size, subset_size=None, to_tensor=False, shuffle=True):
    if data_file.endswith('.npz'):  # allow for labels
        data_dict = np.load(data_file)
        data = data_dict['x']
        if 'y' in data_dict.keys():
            targets = data_dict['y']
            if len(targets.shape) > 1:
                targets = np.squeeze(targets)
                assert len(targets.shape) == 1, f'need target vector. shape is {targets.shape}'
        else:
            targets = None

        if subset_size is not None:
            random_subset = np.random.permutation(data_dict['x'].shape[0])[:subset_size]
            data = data[random_subset]
            targets = targets[random_subset] if targets is not None else None
        synth_data = SynthDataset(data=data, targets=targets, to_tensor=to_tensor)
    else:  # old version
        data = np.load(data_file)
        if subset_size is not None:
            data = data[np.random.permutation(data.shape[0])[:subset_size]]
        synth_data = SynthDataset(data, targets=None, to_tensor=False)

    synth_dataloader = pt.utils.data.DataLoader(synth_data, batch_size=batch_size, shuffle=shuffle,
                                                drop_last=False, num_workers=1)
    return synth_dataloader


class SynthDataset(pt.utils.data.Dataset):
    def __init__(self, data, targets, to_tensor):
        self.labeled = targets is not None
        self.data = data
        self.targets = targets
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = pt.tensor(self.data[idx], dtype=pt.float32) if self.to_tensor else self.data[idx]
        if self.labeled:
            t = pt.tensor(self.targets[idx], dtype=pt.long) if self.to_tensor else self.targets[idx]
            return d, t
        else:
            return d


def mnist_transforms(is_train):
    # if model == 'vgg15':
    #   transform_train = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #   ])
    #   transform_test = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #   ])

    # elif model == 'resnet18':
    transform_train = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.5], [0.5])
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.5], [0.5])
    ])
    # else:
    #   raise ValueError
    return transform_train if is_train else transform_test


def small_data_loader(dataset_name, is_train):
    assert dataset_name in {'fmnist', 'dmnist', 'svhn', 'cifar10_pretrain'}

    if dataset_name in {'fmnist', 'dmnist'}:
        transform = mnist_transforms(is_train)
    else:
        transform = transforms.Compose([
            transforms.CenterCrop((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    root = '../data/'
    if dataset_name == 'dmnist':
        return dset.MNIST(root, train=is_train, transform=transform, download=True)
    elif dataset_name == 'fmnist':
        return dset.FashionMNIST(root, train=is_train, transform=transform, download=True)

    elif dataset_name == 'svhn':
        svhn_dir = os.path.join(root, 'svhn')
        os.makedirs(svhn_dir, exist_ok=True)
        train_str = 'train' if is_train else 'test'
        return dset.SVHN(svhn_dir, split=train_str, transform=transform, download=True)
    elif dataset_name == 'cifar10_pretrain':
        return dset.CIFAR10(root=root, train=is_train, transform=transform, download=True)
    else:
        raise ValueError
