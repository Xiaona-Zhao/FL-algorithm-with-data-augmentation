# foundational function for federated learning algorithm.
import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from Dissertation_code.Create_Dataset import FL_dataset, non_iid_alpha


def FedAvg_mnist_noniid(dataset, num_users):
    num_shards, num_images = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_user = {i: np.array([]) for i in range(num_users)}
    indexs = np.arange(num_shards * num_images)
    labels = dataset.train_labels.numpy()

    indexs_labels = np.vstack((indexs, labels))
    indexs_labels = indexs_labels[:, indexs_labels[1, :].argsort()]
    indexs = indexs_labels[0, :]

    for i in range(num_users):
        random_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - random_set)
        for rand in random_set:
            dict_user[i] = np.concatenate((dict_user[i], indexs[rand * num_images:(rand + 1) * num_images]), axis=0)
    return dict_user


def FedAvg_mnist_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def FedAvg_cifar_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def FedAvg_cifar_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNCifar100(nn.Module):
    def __init__(self):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2704)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 13 * 13, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_dataset(dataset, datatype, iid, non_iid_alpha, world_size):
    global train_dataset, test_dataset, user_groups
    if datatype == 'original FL':
        if dataset == 'mnist':
            data_dir = '../dataset/mnist/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(64), transforms.Normalize((0.1307,), (0.3081,))])
            if iid is True:
                train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
                test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
                user_groups = FedAvg_mnist_iid(train_dataset, world_size)
            elif iid is not True:
                train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
                test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
                user_groups = FedAvg_mnist_noniid(train_dataset, world_size)

        elif dataset == 'femnist':
            data_dir = '../dataset/femnist/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(64), transforms.Normalize((0.1307,), (0.3081,))])
            if iid is True:
                train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
                test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)
                user_groups = FedAvg_mnist_iid(train_dataset, world_size)
            elif iid is not True:
                train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
                test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)
                user_groups = FedAvg_mnist_noniid(train_dataset, world_size)

        elif dataset == 'cifar10':
            data_dir = '../dataset/cifar10/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(64), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            if iid is True:
                train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
                test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
                user_groups = FedAvg_cifar_iid(train_dataset, world_size)
            elif iid is not True:
                train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
                test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
                user_groups = FedAvg_cifar_noniid(train_dataset, world_size)

        elif dataset == 'cifar100':
            data_dir = '../dataset/cifar100/'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(64), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            if iid is True:
                train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
                test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
                user_groups = FedAvg_cifar_iid(train_dataset, world_size)
            elif iid is not True:
                train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
                test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
                user_groups = FedAvg_cifar_noniid(train_dataset, world_size)

    elif datatype == 'non_iid_alpha FL':
        if dataset == 'mnist':
            train_dataset, test_dataset, user_groups = FL_dataset(world_size=world_size, dataset=dataset,
                                                                  non_iid_alpha=non_iid_alpha)
        elif dataset == 'cifar10':
            train_dataset, test_dataset, user_groups = FL_dataset(world_size=world_size, dataset=dataset,
                                                                  non_iid_alpha=non_iid_alpha)
        elif dataset == 'femnist':
            train_dataset, test_dataset, user_groups = FL_dataset(world_size=world_size, dataset=dataset,
                                                                  non_iid_alpha=non_iid_alpha)
        elif dataset == 'cifar100':
            train_dataset, test_dataset, user_groups = FL_dataset(world_size=world_size, dataset=dataset,
                                                                  non_iid_alpha=non_iid_alpha)
    return train_dataset, test_dataset, user_groups


def get_model(dataset, train_dataset):
    global global_model
    img_size = train_dataset[0][0].shape
    len_in = 1
    for x in img_size:
        len_in *= x
        if dataset == 'mnist':
            global_model = CNNMnist()
        elif dataset == 'cifar10':
            global_model = CNNCifar10()
        elif dataset == 'cifar100':
            global_model = CNNCifar100()
    return global_model


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def FedAvg_loader(dataset, idxs):
    idxs_train = idxs[:int(0.8 * len(idxs))]
    idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
    idxs_test = idxs[int(0.9 * len(idxs)):]

    trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=50, shuffle=True)
    validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val) / 10), shuffle=False)
    testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test) / 10), shuffle=False)
    return trainloader, validloader, testloader


class LocalUpdate(object):
    def __init__(self, args, trainloader, validloader, testloader, logger):
        self.args = args
        self.logger = logger
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        client_loss = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # update local model
        for iter in range(self.args.local_epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images), len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                    # record client loss
                    client_loss.append(loss.item())
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            # record batch loss
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), client_loss

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss


def test_inference(model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss
