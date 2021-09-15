
# Implementation for FedAvg
import copy
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse

from FedAVG import get_dataset, get_model, LocalUpdate, average_weights, test_inference


def args_parser():
    parser = argparse.ArgumentParser()

    # Critical parameters for Federated Learning
    parser.add_argument('--epochs', type=int, default=5, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--non_iid_alpha', type=int, default=100, help="Non_iid_alpha for FL")

    # Hyper parameters for Federated Learning
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--local_epoch', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_batch_size', type=int, default=50, help="local batch size: B")
    parser.add_argument('--print_every', type=int, default=1, help="every training epoch to print loss value")
    parser.add_argument('--client_print_every', type=int, default=1, help="every training epoch to print loss value for single client")
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
train_dataset, test_dataset, user_groups = get_dataset(dataset='cifar10', datatype='non_iid_alpha FL', iid=True, non_iid_alpha=args.non_iid_alpha, world_size=10)
# train_dataset, test_dataset, user_groups = get_dataset(dataset='cifar10', datatype='original FL', iid=True, non_iid_alpha=args.non_iid_alpha, world_size=10)


# Get model
global_model = get_model(dataset='cifar10', train_dataset=train_dataset)


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Initialize global weight
global_model.to(device)
global_model.train()
global_weights = global_model.state_dict()


# Initialize parameters
TRAIN_LOSS, TRAIN_ACCURACY, TEST_LOSS, TEST_ACC = [], [], [], []
single_client_loss, single_client_acc = [], []


# Define Federated Training function
def federated_training(args, train_dataset, test_dataset, user_groups):
    # Training model
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n| Global Training Round : {epoch + 1} |')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            if idx == args.specific_client:
                beta = True
            else:
                beta = False
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, loss, local_loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch,beta=beta)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            if idx == args.specific_client:
                single_client_loss.append(local_loss)

        # Update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # Calculate training loss
        loss_avg = sum(local_losses) / len(local_losses)
        TRAIN_LOSS.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
            if c == args.specific_client:
                single_client_acc.append(acc)
        TRAIN_ACCURACY.append(sum(list_acc) / len(list_acc))

        # Print global training loss after every 'print_every' rounds
        if (epoch + 1) % args.print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(TRAIN_LOSS))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * TRAIN_ACCURACY[-1]))
            test_acc, test_loss = test_inference(global_model, test_dataset)
            TEST_LOSS.append(test_loss)
            TEST_ACC.append(test_acc)

    # Test inference after completion of training
    test_acc_final, test_loss_final = test_inference(global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * TRAIN_ACCURACY[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_loss_final))

    return TRAIN_LOSS, TRAIN_ACCURACY, TEST_LOSS, TEST_ACC


# Process Federated Training process
Train_loss, Train_accuracy, Test_loss, Test_accuracy = federated_training(args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups)


# Define Visualization function for loss value
def visualization(args, Train_loss, Train_accuracy, Test_loss, Test_accuracy):
    plt.figure()
    plt.title(f'Training Loss vs Communication rounds under non_iid_alpha = {args.non_iid_alpha}')
    plt.plot(range(len(Train_loss)), Train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    # plt.savefig('Training_loss_alpha_10_mnist.png')
    # plt.show()
    # plt.close()


    plt.figure()
    plt.title(f'Training accuracy vs Communication rounds under non_iid_alpha = {args.non_iid_alpha}')
    plt.plot(range(len(Train_accuracy)), Train_accuracy, color='r')
    plt.ylabel('Training accuracy')
    plt.xlabel('Communication Rounds')
    # plt.savefig('Training_acc_alpha_10_mnist.png')
    # plt.show()
    # plt.close()

    print(f'training_loss = {Train_loss}')
    print(f'train_accuracy = {Train_accuracy}')
    print(f'test_loss = {Test_loss}')
    print(f'test_acc = {Test_accuracy}')

    for item in range(len(single_client_loss)):
        if item % args.client_print_every == 0:
            print(f'Specific training loss for client {args.client_print_every} = {single_client_loss[item]}, global round = {item}')
    print(f'Specific training accuracy for client {args.client_print_every} = {single_client_acc}')


# Process Visualization function
visualization(args, Train_loss=Train_loss, Train_accuracy=Train_accuracy, Test_loss=Test_loss, Test_accuracy=Test_accuracy)

