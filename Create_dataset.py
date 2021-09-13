from Data_generation.Data_prepare import get_dataset
from Data_generation.Data_partition import partition_indices
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import random
import time


# Highlight: when the non_iid_alpha == 0.01 and world_size > 20, partition time will be extremely high


# define random non_iid_alpha function
def non_iid_alpha(num_round):
    col = []
    # 将0-1的随机数，仿射变换放大到0.01 - 100，通过线性变换和指数函数
    for i in range(num_round):
        init_a = random.random()
        init_b = 4 * init_a - 2
        init_c = 10 ** init_b
        col.append(init_c)
    return col


# define transformation for training dataset
# def Data_transform(data_name):
#     global Transform
#     if data_name == 'cifar10':
#         Transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
#                                          transforms.Normalize([0.5 for _ in range(3)],
#                                                               [0.5 for _ in range(3)])])
#     elif data_name == 'mnist':
#         Transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])
#     return Transform


# define training dataset under Federated Learning mode
def FL_dataset(world_size, dataset, non_iid_alpha):
    global data_train, data_test
    world_size = world_size
    n_workers = [1.0 / world_size for _ in range(world_size)]

    if dataset == 'mnist':
        data_train = get_dataset(name='mnist', datasets_path='../dataset', is_train=True, download=True)
        data_test = get_dataset(name='mnist', datasets_path='../dataset', is_train=False, download=True)
    elif dataset == 'cifar10':
        data_train = get_dataset(name='cifar10', datasets_path='../dataset', is_train=True, download=True)
        data_test = get_dataset(name='cifar10', datasets_path='../dataset', is_train=False, download=True)

    # partition indices using non_iid_alpha
    partitioned_indices = partition_indices(data=data_train, partition_type='non_iid_dirichlet',
                                            partition_count=n_workers, non_iid_alpha=non_iid_alpha)

    return data_train, data_test, partitioned_indices


def demo():
    start_time = time.time()
    alpha = 100
    data_train, data_test, partitioned_indices = FL_dataset(world_size=10, dataset='cifar10', non_iid_alpha=alpha)
    end_time = time.time()
    train_loader = DataLoader(data_train, batch_size=100, shuffle=True)
    for batch_idx, (real, _) in enumerate(train_loader):
        print(real.shape)
        break
    print(f'time = {end_time - start_time} seconds')
    return data_train, data_test, partitioned_indices


# demo()


# 一次alpha值进行partition
# data_train, data_test, partitioned_indices = demo()
# print(len(data_train))
# print(len(partitioned_indices))

# 多次alpha值进行partition
# alpha = non_iid_alpha(num_round=10)
# client_0_indices = []
# for item in range(len(alpha)):
#     data_train, data_test, partitioned_indices = FL_dataset(world_size=10, dataset='mnist', non_iid_alpha=alpha[item])
#     print(data_test.targets[0:20])
#     client_0_indices.append(partitioned_indices[0])
# print(client_0_indices)

# 将以data.number（0-60000）为格式的partition_index改变为data.targets格式[0-10]
# def partition_dataidx(partitions):
#     targets = data_train.targets
#     n_workers = 10
#     num_classes = len(np.unique(data_train.targets))
#     shuffle_dataidx = {}
#     for i in range(n_workers):
#         np.random.shuffle(partitions[i])
#         shuffle_dataidx[i] = partitions[i]
#     partitioned_dataidx = {}
#     for i in shuffle_dataidx:
#         partitioned_dataidx[i] = {}
#         list = []
#         for item in shuffle_dataidx[i]:
#             list.append(targets[item])
#         l = np.array(list)
#         list_2 = []
#         for j in range(num_classes):
#             # print("client-" + str(i) + ",label-" + str(j), np.where(l == j)[0].size / l.size)
#             list_2.append(np.where(l == j)[0].size / l.size)
#         partitioned_dataidx[i] = list_2
#     return partitioned_dataidx

# partitioned_dataidx = partition_dataidx(partitioned_indices)
# print(len(partitioned_dataidx))
# print(partitioned_dataidx)





