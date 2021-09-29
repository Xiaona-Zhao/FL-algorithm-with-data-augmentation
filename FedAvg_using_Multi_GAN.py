# Implementation for FedAvg
import copy
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
import torch.utils.data as Data
import matplotlib.pyplot as plt
import argparse

from FedAVG import get_dataset, get_model, LocalUpdate, average_weights, test_inference
from Augmentation_using_Multi_GAN import GAN_net, GAN_optimizer, Disc_train, Gen_train, Multi_disc_training


def args_parser():
    parser = argparse.ArgumentParser()

    # Critical parameters for Federated Learning
    parser.add_argument('--epochs', type=int, default=5, help="number of rounds of FL training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--non_iid_alpha', type=int, default=100, help="Non_iid_alpha for FL")

    # Hyper parameters for Multi_GAN training
    parser.add_argument('--Gan_lr', type=int, default=2e-4, help="Learning rate of Multi_GAN training")
    parser.add_argument('--Gan_epochs', type=int, default=50, help="number of rounds of Multi_GAN training")
    parser.add_argument('--Generator_paths', type=int, default=1, help="Generator_paths for Multi_GAN")
    parser.add_argument('--num_discriminator', type=int, default=10, help="num_discriminator for Multi_GAN")
    parser.add_argument('--classifier_parameter', type=int, default=0, help="Hyper-parameter for classifier")
    parser.add_argument('--optimizer_type', type=str, default='Adam', help="optimizer_type for Multi_GAN")
    parser.add_argument('--Channels_Num', type=int, default=1, help="number of Channels for input image")
    parser.add_argument('--Channel_Noise', type=int, default=128, help="number of Channels for input noise")
    parser.add_argument('--Batch_size', type=int, default=50, help="Batch size for Gan training process. This batch_size equals to local_batch_size in Federated learning")

    # Hyper parameters for Federated Learning
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--local_epoch', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_batch_size', type=int, default=50, help="local batch size: B")
    parser.add_argument('--print_every', type=int, default=1, help="every training epoch to print loss value")
    parser.add_argument('--client_print_every', type=int, default=1,
                        help="every training epoch to print loss value for single client")
    parser.add_argument('--specific_client', type=int, default=1, help="the specific client chosen to print loss value")

    # Hyper parameters for neural networks
    parser.add_argument('--optimizer', type=str, default='adam', help="type of optimizer")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0002, help='Adan weight decay (default: 2e-4)')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    args = parser.parse_args()
    return args


args = args_parser()
logger = SummaryWriter('../logs')

# Get dataset
train_dataset, test_dataset, user_groups = get_dataset(dataset='mnist', datatype='non_iid_alpha FL', iid=True,
                                                       non_iid_alpha=args.non_iid_alpha, world_size=10)
# train_dataset, test_dataset, user_groups = get_dataset(dataset='cifar10', datatype='original FL', iid=True, non_iid_alpha=args.non_iid_alpha, world_size=10)


# Get model
global_model = get_model(dataset='mnist', train_dataset=train_dataset)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize global weight
global_model.to(device)
global_model.train()
global_weights = global_model.state_dict()

# Initialize parameters
TRAIN_LOSS, TRAIN_ACCURACY, TEST_LOSS, TEST_ACC = [], [], [], []
single_client_loss, single_client_acc = [], []


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def train_val_test(args, dataset, idxs):
    # split indexes for train, validation, and test (80, 10, 10)
    idxs_train = idxs[:int(0.8 * len(idxs))]
    idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
    idxs_test = idxs[int(0.9 * len(idxs)):]

    # train_dataset = DatasetSplit(dataset, idxs_train)
    # valid_dataset = DatasetSplit(dataset, idxs_val)
    # test_dataset = DatasetSplit(dataset, idxs_test)

    train_loader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=args.local_batch_size, shuffle=True)
    valid_loader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val) / 10), shuffle=False)
    test_loader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test) / 10), shuffle=False)
    return train_loader, valid_loader, test_loader
    # return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader


Generator, Generator_single_path, Discriminator_list = GAN_net(args, GAN_type='MPI_GAN')
opt_gen, opt_disc_list = GAN_optimizer(args, Generator=Generator, Discriminator_list=Discriminator_list)


def federated_training(args, train_dataset, test_dataset, user_groups):
    # Training model
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n| Global Training Round : {epoch + 1} |')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        if epoch == 0:
            Seged_dataset = []
            Augmentation_dataset = Multi_disc_training(args, train_dataset=train_dataset, idxs_users=idxs_users, user_groups=user_groups, Generator=Generator, Discriminator_list=Discriminator_list,
                                                           opt_gen=opt_gen, opt_disc_list=opt_disc_list)
            # length = len(Augmentation_dataset)
            # Seged_dataset_size = int(length/args.num_users)
            # for i in range(args.num_users):
            #     Seged_dataset.append(torch.utils.data.random_split(Augmentation_dataset, [Seged_dataset_size]))
            # print(Seged_dataset)

        # 定义本次client号码，及其client数据
        for idx in idxs_users:
            # 将train_loader, valid_loader, test_loade全局化
            train_loader, valid_loader, test_loader = train_val_test(args, train_dataset, list(user_groups[idx]))

            if idx == args.specific_client:
                beta = True
            else:
                beta = False
            # local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            local_model = LocalUpdate(args=args, train_loader=train_loader, test_loader=test_loader, logger=logger)
            weight, loss, local_loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, beta=beta)
            local_weights.append(copy.deepcopy(weight))
            local_losses.append(copy.deepcopy(loss))
            if idx == args.specific_client:
                single_client_loss.append(local_loss)

        # Update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # Calculate training loss
        loss_avg = sum(local_losses) / len(local_losses)
        TRAIN_LOSS.append(loss_avg)

        # Print global training loss after every 'print_every' rounds
        if (epoch + 1) % args.print_every == 0:
            print(f' \nSystem Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.array(TRAIN_LOSS[-1])}')
            test_acc, test_loss = test_inference(global_model, test_dataset)
            TEST_LOSS.append(test_loss)
            TEST_ACC.append(test_acc)

    # Test inference after completion of training
    test_acc_final, test_loss_final = test_inference(global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Loss: {:.2f}%".format(test_loss_final))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc_final))

    return TRAIN_LOSS, TEST_LOSS, TEST_ACC


# Process Federated Training process
Train_loss, Test_loss, Test_accuracy = federated_training(args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups)


# Define Visualization function for loss value
def visualization(args, Train_loss, Test_loss, Test_accuracy):
    plt.figure()
    plt.title(f'Training Loss vs Communication rounds under non_iid_alpha = {args.non_iid_alpha}')
    plt.plot(range(len(Train_loss)), Train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    # plt.savefig('Training_loss_alpha_10_mnist.png')
    # plt.show()
    # plt.close()

    print(f'training_loss = {Train_loss}')
    print(f'test_loss = {Test_loss}')
    print(f'test_acc = {Test_accuracy}')

    for item in range(len(single_client_loss)):
        if item % args.client_print_every == 0:
            print(
                f'Specific training loss for client {args.client_print_every} = {single_client_loss[item]}, global round = {item}')
    print(f'Specific training accuracy for client {args.client_print_every} = {single_client_acc}')


# Process Visualization function
# visualization(args, Train_loss=Train_loss, Test_loss=Test_loss, Test_accuracy=Test_accuracy)
