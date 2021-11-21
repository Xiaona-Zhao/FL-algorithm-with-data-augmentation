import copy
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse

from Dissertation_code.Functions_Data_Augmentation import GAN_net, GAN_optimizer
from Dissertation_code.Functions_Fed_Avg import get_dataset, get_model, test_inference, average_weights, LocalUpdate, \
    DatasetSplit
from Dissertation_code.Data_Augmentation import Get_generator_model, Get_classifier_model, Get_augmentation_dataset


# split the index of training dataset to 8:1:1
def train_val_test(idxs):
    idxs_train = idxs[:int(0.8 * len(idxs))]
    idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
    idxs_test = idxs[int(0.9 * len(idxs)):]
    return idxs_train, idxs_val, idxs_test


# main function of FL algorithm
def Federated_learning(args, using_augmentation, FL_type, iid):
    global user_groups, train_dataset, Augmentation_dataset
    logger = SummaryWriter('../logs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get partitioned dataset
    if FL_type == 'origin':
        train_dataset, test_dataset, user_groups = get_dataset(dataset=args.dataset, datatype='original FL', iid=iid,
                                                               non_iid_alpha=None, world_size=args.num_users)
    elif FL_type == 'non_iid_alpha':
        train_dataset, test_dataset, user_groups = get_dataset(dataset=args.dataset, datatype='non_iid_alpha FL',
                                                               iid=None, non_iid_alpha=args.non_iid_alpha,
                                                               world_size=args.num_users)

    # get global model to train FL algorithm
    global_model = get_model(dataset=args.dataset, train_dataset=train_dataset)
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()

    # initialize parameters
    train_loss, train_accuracy, test_loss, test_acc = [], [], [], []
    single_client_loss, single_client_acc = [], []

    # get augmentation dataset based on partitioned data for federated learning algorithm
    if using_augmentation is True:
        # get GAN model
        Generator, Discriminator_list = GAN_net(args, GAN_type=args.Gan_type)
        opt_gen, opt_disc_list = GAN_optimizer(args, Generator=Generator, Discriminator_list=Discriminator_list)

        # get generator and classifier model for data augmentation
        generator = Get_generator_model(args=args, train_dataset=train_dataset, user_groups=user_groups,
                                        Generator=Generator, Discriminator_list=Discriminator_list, opt_gen=opt_gen,
                                        opt_disc_list=opt_disc_list)
        classifier = Get_classifier_model(args=args)

        # get augmentation dataset
        Augmentation_dataset = Get_augmentation_dataset(args=args, Generator=generator, Classifier=classifier,
                                                        Augmentation_size=12)
        print('This FL training process is data augmented using Multi_path GAN')

    # train federated learning
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        # get index user
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            # get splited dataset for single user
            idxs_train, idxs_val, idxs_test = train_val_test(list(user_groups[idx]))
            traindataset = DatasetSplit(train_dataset, idxs_train)
            validdataset = DatasetSplit(train_dataset, idxs_val)
            testdataset = DatasetSplit(train_dataset, idxs_test)

            # get Local_model for federated learning based on using augmentataion or not
            if using_augmentation is True:
                # concat raw dataset and augmentation dataset
                Augmented_dataset = torch.utils.data.ConcatDataset([Augmentation_dataset, traindataset])
                index = []
                for i in range(len(Augmented_dataset)):
                    index.append(i)

                # simplify dataset for training
                Augmented_dataset_simplified = DatasetSplit(Augmented_dataset, index)

                # get train_loader
                augmentedloader = DataLoader(Augmented_dataset_simplified, batch_size=50, shuffle=True)
                validloader = DataLoader(validdataset, batch_size=6, shuffle=False)
                testloader = DataLoader(testdataset, batch_size=6, shuffle=False)

                # get local model for augmented dataset
                local_model = LocalUpdate(args=args, trainloader=augmentedloader, validloader=validloader,
                                          testloader=testloader, logger=logger)  # using_augmentation is True

            elif using_augmentation is False:
                # without augmentation
                trainloader = DataLoader(traindataset, batch_size=50, shuffle=True)
                validloader = DataLoader(validdataset, batch_size=6, shuffle=False)
                testloader = DataLoader(testdataset, batch_size=6, shuffle=False)
                local_model = LocalUpdate(args=args, trainloader=trainloader, validloader=validloader,
                                          testloader=testloader, logger=logger)

            # train on FedAvg algorithm, record local loss for specific client
            w, loss, client_loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            if idx == args.specific_client:
                single_client_loss.append(client_loss)

        # global aggregation
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # record train_accuracy
        # list_acc, list_loss = [], []
        # global_model.eval()
        # for c in range(args.num_users):
        #     # local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
        #     local_model = LocalUpdate_aug(args=args, trainloader=trainloader, validloader=validloader, testloader=testloader, logger=logger)
        #     acc, loss = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc) / len(list_acc))

        # record test_acc and test_loss
        test_acc_temp, test_loss_temp = test_inference(global_model, test_dataset)
        test_acc.append(test_acc_temp)
        test_loss.append(test_loss_temp)
    return train_loss, train_accuracy, test_loss, test_acc, single_client_loss


# record loss and accuracy value
def visualization(Train_loss, Train_accuracy, Test_loss, Test_accuracy, single_client_loss):
    np.save('%s/Train_loss.npy' % './results/log_file', Train_loss)
    np.save('%s/Train_accuracy.npy' % './results/log_file', Train_accuracy)
    np.save('%s/Test_loss.npy' % './results/log_file', Test_loss)
    np.save('%s/Test_accuracy.npy' % './results/log_file', Test_accuracy)
    print(f'training_loss = {Train_loss}')
    print(f'training_acc = {Train_accuracy}')
    print(f'test_loss = {Test_loss}')
    print(f'test_acc = {Test_accuracy}')
    print(f'single_client_loss for client {args.specific_client} = {single_client_loss}')


def args_parser():
    parser = argparse.ArgumentParser()

    # Critical parameters for Federated Learning
    parser.add_argument('--dataset', type=str, default='mnist',
                        help="select training dataset: mnist, femnist, cifar10 and cifar100")
    parser.add_argument('--epochs', type=int, default=500, help="number of epochs for FL training")
    parser.add_argument('--num_users', type=int, default=100, help="number of clients for FL training")
    parser.add_argument('--non_iid_alpha', type=int, default=100, help="Non_iid_alpha for FL training")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')

    # Hyper parameters for Multi_GAN training
    parser.add_argument('--Gan_type', type=str, default='mnist_GAN',
                        help="select model for Multi_GAN training: mnist_GAN, cifar10_GAN of DCGAN")
    parser.add_argument('--Gan_lr', type=int, default=2e-4, help="Learning rate for Multi_GAN training")
    parser.add_argument('--Gan_epochs', type=int, default=50, help="Number of epochs for Multi_GAN training")
    parser.add_argument('--Generator_paths', type=int, default=4, help="Generator_paths for multi_path generator")
    parser.add_argument('--num_discriminator', type=int, default=100,
                        help="Number of discriminators, should be corresponding to the number of clients")
    parser.add_argument('--classifier_parameter', type=int, default=0, help="Hyper-parameter for classifier")
    parser.add_argument('--optimizer_type', type=str, default='Adam',
                        help="Optimizer for GAN training, Adam or Adagrad")

    parser.add_argument('--Channels_Num', type=int, default=1, help="number of Channels for input image")
    parser.add_argument('--Channel_Noise', type=int, default=128, help="number of Channels for input noise")
    parser.add_argument('--Training_Batch_size', type=int, default=50,
                        help="Batch size for Gan training process. This batch_size equals to local_batch_size in Federated learning")
    parser.add_argument('--Testing_Batch_size', type=int, default=20, help="Batch size for Gan generation process")
    parser.add_argument('--visualization', type=int, default=0,
                        help="visualization of generated images in Multi_path generator")

    # Hyper parameters for Federated Learning
    parser.add_argument('--local_epoch', type=int, default=5, help="The number of local epochs: E")
    parser.add_argument('--local_batch_size', type=int, default=50, help="Local batch size: B")
    parser.add_argument('--client_print_every', type=int, default=1,
                        help="Every training epoch to print loss value for single client")
    parser.add_argument('--specific_client', type=int, default=1,
                        help="The specific client chosen to print local loss value")

    # Hyper parameters for neural networks
    parser.add_argument('--optimizer', type=str, default='adam', help="optimizer for FL training")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for FL training')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0002, help='Adan weight decay (default: 2e-4)')
    parser.add_argument('--verbose', type=int, default=1, help='print local training details in FL programme')
    args = parser.parse_args()
    return args


args = args_parser()

train_loss, train_accuracy, test_loss, test_acc, single_client_loss = Federated_learning(args, using_augmentation=True,
                                                                                         FL_type='non_iid_alpha',
                                                                                         iid=None)
visualization(Train_loss=train_loss, Train_accuracy=train_accuracy, Test_loss=test_loss, Test_accuracy=test_acc,
              single_client_loss=single_client_loss)
