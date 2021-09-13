import os
import torchvision.datasets as dataset
import torchvision.transforms as transforms


Image_size = 64


def _get_mnist(root, is_train, download):
    normalize = (transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose([transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    return dataset.MNIST(root=root, train=is_train, transform=transform, download=download,)


def _get_cifar10(root, is_train, download):
    normalize = (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose([transforms.Resize(Image_size), transforms.ToTensor()]
            + ([normalize] if normalize is not None else []))

    return dataset.CIFAR10(root=root, train=is_train, transform=transform, download=download)


def get_dataset(name, datasets_path, is_train, download=True):
    root = os.path.join(datasets_path, name)
    if name == "mnist":
        return _get_mnist(root=root, is_train=is_train, download=download)
    elif name == 'cifar10':
        return _get_cifar10(root=root, is_train=is_train, download=download)
    else:
        raise NotImplementedError

import numpy as np
# mnist_dataset = get_dataset(name='mnist', datasets_path='../data', is_train=True, download=True)
# cifar_dataset = get_dataset(name='cifar10', datasets_path='../data', is_train=True, download=True)
# print(np.unique(cifar_dataset.targets))
# print(cifar_dataset.data)

