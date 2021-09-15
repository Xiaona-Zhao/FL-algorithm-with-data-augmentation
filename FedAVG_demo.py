# A really simple version of FedAvg using MLP and MNIST as self training
import random
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
# device = 'cuda' if args.gpu else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# cifar_dataset = datasets.CIFAR10(root='../dataset/cifar/', train=True, download=True,)
# print(len(cifar_dataset.targets))
# mnist_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True,)
# print(len(mnist_dataset.train_labels))


# from Create_dataset import FL_dataset, non_iid_alpha
from Data_generation.Create_dataset import FL_dataset, non_iid_alpha

TEST_LOSS = []
TEST_ACC = []

def MNIST_noniid(dataset, num_users):
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
            dict_user[i] = np.concatenate(
                (dict_user[i], indexs[rand * num_images:(rand + 1) * num_images]), axis=0
            )

    return dict_user


def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
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
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


# heperparameter
def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args


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


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 训练模型，载入hyperparameter和dataset
args = args_parser()
logger = SummaryWriter('../logs')

#
# utils_1 - get train / test dataset and user_groups
def get_dataset(args):
    if args.dataset == 'mnist':
        data_dir = '../dataset/mnist/'
        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        user_groups = mnist_iid(train_dataset, args.num_users)

    elif args.dataset == 'cifar':
        data_dir = '../dataset/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
        user_groups = cifar_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


# train_dataset, test_dataset, user_groups = get_dataset(args)


# utils_2 - average weight
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


# 提供data载入功能，变量为dataset 和 train/test 的 idxs
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# 主要包括update_weight功能
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val) / 10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, beta):
        # Set mode to train model
        model.train()
        epoch_loss = []
        local_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=2e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1000 == 0) and beta == True:
                    # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     global_round,
                    #     iter,
                    #     batch_idx * len(images),
                    #     len(self.trainloader.dataset),
                    #     100. * batch_idx / len(self.trainloader), loss.item()))
                    local_loss.append(loss.item())
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), local_loss

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss


# 提供test的accuracy 和loss数值，不包含training部分
def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


world_size = 10
user_groups = []
alpha = non_iid_alpha(num_round=args.epochs)
# alpha = []
# for i in range(args.epochs):
#     alpha.append(10)

for item in range(len(alpha)):
    # print(alpha[item])
    _, _, User_g = FL_dataset(world_size=world_size, dataset='mnist', non_iid_alpha=alpha[item])
    user_groups.append(User_g)

train_dataset, test_dataset, _ = FL_dataset(world_size=world_size, dataset='mnist', non_iid_alpha=100)



# 训练模型，fit model
img_size = train_dataset[0][0].shape
len_in = 1
for x in img_size:
    len_in *= x
    global_model = CNNMnist(args)
# if args.gpu:
#     torch.cuda.set_device(args.gpu)




global_model.to(device)
global_model.train()
# print(global_model)

# 初始化global weight
global_weights = global_model.state_dict()
# print(global_weights)

# 初始化参数
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 1
val_loss_pre, counter = 0, 0










single_client_loss = []
single_client_acc = []

# 训练模型
for epoch in tqdm(range(args.epochs)):
    local_weights, local_losses = [], []
    # 打印global round
    print(f'\n | Global Training Round : {epoch + 1} |\n')

    global_model.train()
    # 取frac*num_users乘积，赋值为m
    m = max(int(args.frac * args.num_users), 1)
    # 在num_users中随机取出m个user 作为index_user
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # 每个被选中的idx_user都跑一遍local_model,同时更新local_weight和local_losses
    # LOCAL ！！！
    for idx in idxs_users:
        if idx == 1:
            beta = True
        else:
            beta = False
        local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[epoch][idx], logger=logger)
        # local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
        w, loss, local_loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, beta=beta)
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))
        if idx == 1:
            single_client_loss.append(local_loss)
            # print(idx)
            # print(single_client_loss)

    # update global weights
    global_weights = average_weights(local_weights)

    # global_weights 重新赋值global_model，重新训练
    global_model.load_state_dict(global_weights)

    # 计算training loss
    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)

    # Calculate avg training accuracy over all users at every epoch
    # 计算training accuracy
    list_acc, list_loss = [], []
    global_model.eval()
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[epoch][idx], logger=logger)
        # local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
        acc, loss = local_model.inference(model=global_model)
        list_acc.append(acc)
        list_loss.append(loss)
        # if c == 1:
        #     single_client_acc.append(acc)
    train_accuracy.append(sum(list_acc) / len(list_acc))

    # print global training loss after every 'i' rounds
    if (epoch + 1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        TEST_LOSS.append(test_loss)
        TEST_ACC.append(test_acc)





# Test inference after completion of training
test_acc_final, test_loss_final = test_inference(args, global_model, test_dataset)

print(f' \n Results after {args.epochs} global rounds of training:')
print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
print("|---- Test Accuracy: {:.2f}%".format(100 * test_loss_final))



plt.figure()
# plt.title('Training Loss vs Communication rounds under non_iid_alpha = 10')
plt.title('Training Loss vs Communication rounds origin data iid')
plt.plot(range(len(train_loss)), train_loss, color='r')
plt.ylabel('Training loss')
plt.xlabel('Communication Rounds')
plt.savefig('Training_loss_origin_iid_mnist')
# plt.savefig('Training_loss_alpha_10_mnist.png')
plt.show()

plt.figure()
# plt.title('Training accuracy vs Communication rounds under non_iid_alpha = 10')
plt.title('Training accuracy vs Communication rounds origin data iid')
plt.plot(range(len(train_accuracy)), train_accuracy, color='r')
plt.ylabel('Training accuracy')
plt.xlabel('Communication Rounds')
plt.savefig('Training_acc_origin_iid_mnist')
# plt.savefig('Training_acc_alpha_10_mnist.png')
plt.show()
print(f'training_loss = {train_loss}')
print(f'train_accuracy = {train_accuracy}')
print(f'test_loss = {TEST_LOSS}')
print(f'test_acc = {TEST_ACC}')


for item in range(len(single_client_loss)):
    if item % 5 == 0:
        print(f'Specific training loss for client 1 = {single_client_loss[item]}, global round = {item}')
# print(f'single acc for client 1 = {single_client_acc}')


