from Dissertation_code.Dataset_prepare import get_dataset
from Dissertation_code.Partition_dataset import partition_indices
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import time


# define random non_iid_alpha function
def non_iid_alpha(num_round):
    col = []
    for i in range(num_round):
        init_a = random.random()
        init_b = 4 * init_a - 2
        init_c = 10 ** init_b
        col.append(init_c)
    return col


# define training dataset under Federated Learning mode
def FL_dataset(world_size, dataset, non_iid_alpha):
    global data_train, data_test
    num_workers = [1.0 / world_size for _ in range(world_size)]

    if dataset == 'mnist':
        data_train = get_dataset(name='mnist', datasets_path='../dataset', is_train=True, download=True)
        data_test = get_dataset(name='mnist', datasets_path='../dataset', is_train=False, download=True)
    elif dataset == 'femnist':
        data_train = get_dataset(name='femnist', datasets_path='../dataset', is_train=True, download=True)
        data_test = get_dataset(name='femnist', datasets_path='../dataset', is_train=False, download=True)
    elif dataset == 'cifar10':
        data_train = get_dataset(name='cifar10', datasets_path='../dataset', is_train=True, download=True)
        data_test = get_dataset(name='cifar10', datasets_path='../dataset', is_train=False, download=True)
    elif dataset == 'cifar100':
        data_train = get_dataset(name='cifar100', datasets_path='../dataset', is_train=True, download=True)
        data_test = get_dataset(name='cifar100', datasets_path='../dataset', is_train=False, download=True)

    # partition indices using non_iid_alpha
    partitioned_indices = partition_indices(data=data_train, partition_count=num_workers, non_iid_alpha=non_iid_alpha)
    return data_train, data_test, partitioned_indices


def visualization_analysis(data_train, data_test, num_workers, partitions):
    targets = data_train.targets
    num_classes = len(np.unique(data_train.targets))
    shuffle_data_index = {}
    for i in range(num_workers):
        np.random.shuffle(partitions[i])
        shuffle_data_index[i] = partitions[i]
    partitioned_dataidx = {}
    for i in shuffle_data_index:
        partitioned_dataidx[i] = {}
        list = []
        for item in shuffle_data_index[i]:
            list.append(targets[item])
        l = np.array(list)
        list_2 = []
        for j in range(num_classes):
            list_2.append(np.where(l == j)[0].size / l.size)
        partitioned_dataidx[i] = list_2
    return partitioned_dataidx


def demo():
    start_time = time.time()
    alpha = 100
    num_clients = 10

    data_train, data_test, partitioned_indices = FL_dataset(world_size=num_clients, dataset='mnist',
                                                            non_iid_alpha=alpha)
    end_time = time.time()

    train_loader = DataLoader(data_train, batch_size=100, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=100, shuffle=False)

    for batch_idx, (real, label) in enumerate(train_loader):
        print(f"training data shape = {real.shape}")
        print(f"data label shape = {label.shape}")
        break

    print(f'time = {end_time - start_time} seconds')

    partitioned_dataidx = visualization_analysis(data_train=data_train, data_test=data_test, num_workers=num_clients,
                                                 partitions=partitioned_indices)
    print(f'partitioned data index = {partitioned_dataidx}')
    return data_train, data_test, partitioned_indices


# demo()
