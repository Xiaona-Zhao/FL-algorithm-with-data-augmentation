import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import argparse
from Module import FC_Generator, FC_Discriminator, DCGAN_Gen, DCGAN_Disc, DCGAN_Gen_Single_Path, \
    initialize_weights, MPI_D, MPI_G, MPI_G_single_path
from Create_dataset import FL_dataset
iter = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
print(f'device = {device}')


# 定义, 初始化GAN网络, One Generator with multi paths, and multi Discriminator corresponding to each FL client dataset
def GAN_net(args, GAN_type):
    global net_G, net_D, net_G_single_path
    if GAN_type == 'DCGAN':
        net_G = DCGAN_Gen(noise_dim=args.Channel_Noise, channels_num=args.Channels_Num, feature_gen=8, G_paths=args.Generator_paths)
        net_G_single_path = DCGAN_Gen(noise_dim=args.Channel_Noise, channels_num=args.Channels_Num, feature_gen=8, G_paths=1)
        net_D = DCGAN_Disc(channels_num=args.Channels_Num, feature_disc=8, G_paths=args.Generator_paths)
        initialize_weights(net_G)
        initialize_weights(net_G_single_path)
        initialize_weights(net_D)
    elif GAN_type == 'MPI_GAN':
        net_G = MPI_G(noise_dim=args.Channel_Noise, G_paths=args.Generator_paths)
        net_G_single_path = MPI_G_single_path(noise_dim=args.Channel_Noise, channels_num=args.Channels_Num, G_paths=1)
        net_D = MPI_D(channels_num=args.Channels_Num, G_paths=args.Generator_paths)
    else:
        print('GAN_type undefined')

    # sent neuron network to device
    Generator = net_G.to(device)
    Generator_single_path = net_G_single_path.to(device)
    Discriminator_list = []
    for i in range(args.num_discriminator):
        Discriminator = net_D.to(device)
        Discriminator_list.append(Discriminator)
    return Generator, Generator_single_path, Discriminator_list


def GAN_optimizer(args, Generator, Discriminator_list):
    opt_gen = []
    opt_disc_list = []
    if args.optimizer_type == 'Adam':
        for i in range(args.Generator_paths):
            opt_gen.append(optim.Adam(Generator.paths[i].parameters(), lr=args.Gan_lr, betas=(0.5, 0.999)))
        for i in range(args.num_discriminator):
            Discriminator = Discriminator_list[i]
            opt_disc = optim.Adam(Discriminator.parameters(), lr=args.Gan_lr, betas=(0.5, 0.999))
            opt_disc_list.append(opt_disc)

    elif args.optimizer_type == 'Adagrad':
        for i in range(args.Generator_paths):
            opt_gen.append(optim.Adagrad(Generator.paths[i].parameters(), lr=args.Gan_lr))
        for i in range(args.num_discriminator):
            Discriminator = Discriminator_list[i]
            opt_disc = optim.Adagrad(Discriminator.parameters(), lr=args.Gan_lr)
            opt_disc_list.append(opt_disc)
    return opt_gen, opt_disc_list


# Loss Function
criterion = nn.BCELoss().to(device)


def Disc_train(args, Generator, Discriminator, real, noise, loss_disc_total):
    # 计算输入为real的DISC_loss
    Discriminator_real, classifier_r = Discriminator(real)
    Discriminator_real = Discriminator_real.reshape(-1)  # Discriminator_real.shape = [50]
    loss_disc_real = criterion(Discriminator_real, torch.ones_like(Discriminator_real))

    loss_classifier_total = 0
    loss_disc_fake_total = 0
    for i in range(args.Generator_paths):
        # 计算输入为fake的DISC_loss
        fake = Generator.paths[i](noise)  # fake.shape = [50, 1, 64, 64]
        Discriminator_fake, classifier_f = Discriminator(fake.detach())

        Discriminator_fake = Discriminator_fake.reshape(-1)  # Discriminator_fake.shape = [50]
        classifier_f = classifier_f.squeeze(-1)  # classifier_f.shape = [50, 1, 1]

        loss_disc_fake = criterion(Discriminator_fake, torch.zeros_like(Discriminator_fake))
        loss_disc_fake_total += loss_disc_fake

        # 将 target进行维度重构
        target = Variable(Tensor(real.size(0)).fill_(i), requires_grad=False)  # target.shape = [128]
        target = target.type(Tensor)
        target = target.unsqueeze(1)  # classifier_f.shape = [50, 1]

        loss_classifier = F.nll_loss(classifier_f, target) * args.classifier_parameter
        loss_classifier_total += loss_classifier

    loss_classifier_avg = loss_classifier_total / args.Generator_paths
    loss_disc_fake_avg = loss_disc_fake_total / args.Generator_paths

    # 将两个loss加和/2为DISC的loss
    loss_disc = (loss_disc_fake_avg + loss_disc_real) / 2
    loss_disc_total += loss_disc + loss_classifier_avg
    return loss_disc_total


def Gen_train(args, Path, Generator, Discriminator, real, noise, loss_gen_total):
    fake = Generator.paths[Path](noise)
    output, classifier = Discriminator(fake)
    # classifier = classifier.view(-1)
    # classifier = classifier.squeeze(-1)

    loss_gen = criterion(output, torch.ones_like(output))

    # target = Variable(Tensor(real.size(0)).fill_(Path), requires_grad=False)
    # target = target.type(Tensor)
    # target = target.unsqueeze(1)
    # loss_classifier = F.nll_loss(classifier, target) * args.classifier_parameter

    loss_gen_total = loss_gen  # + loss_classifier
    return loss_gen_total


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def train_val_test(dataset, idxs):
    # split indexes for train, validation, and test (80, 10, 10)
    idxs_train = idxs[:int(0.8 * len(idxs))]
    idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
    idxs_test = idxs[int(0.9 * len(idxs)):]

    # train_dataset = DatasetSplit(dataset, idxs_train)
    # valid_dataset = DatasetSplit(dataset, idxs_val)
    # test_dataset = DatasetSplit(dataset, idxs_test)

    train_loader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=50, shuffle=True)
    valid_loader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val) / 10), shuffle=False)
    test_loader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test) / 10), shuffle=False)
    return train_loader, valid_loader, test_loader


def Multi_disc_training(args, train_dataset, idxs_users, user_groups, Generator, Discriminator_list, opt_gen, opt_disc_list):
    global Augmentation_dataset
    Generator.train()

    for Discriminator in Discriminator_list:
       Discriminator.train()

    for epoch in range(args.Gan_epochs):
        for idx in idxs_users:
            assert args.num_users == args.num_discriminator
            train_loader, valid_loader, test_loader = train_val_test(train_dataset, list(user_groups[idx]))

            # define train loader for each client, and training in coresponding Discriminator
            for batch_idx, (real, _) in enumerate(train_loader):
                real = real.to(device) #real.shape = [50, 1, 64, 64]
                noise = torch.randn(args.Batch_size, args.Channel_Noise, 1, 1).to(device)

                loss_gen_total = 0
                loss_disc_total = 0

                # training Discriminator
                loss_disc = Disc_train(args, Generator=Generator, Discriminator=Discriminator_list[idx],
                                       real=real, noise=noise, loss_disc_total=loss_disc_total)
                Discriminator_list[idx].zero_grad()
                loss_disc.backward(retain_graph=True)
                opt_disc_list[idx].step()

                # training Generator
                for path in range(args.Generator_paths):
                    loss_gen_total = Gen_train(args, Path=path, Generator=Generator, Discriminator=Discriminator_list[idx],
                                               real=real, noise=noise, loss_gen_total=loss_gen_total)
                    Generator.paths[path].zero_grad()
                    loss_gen_total.backward()
                    opt_gen[path].step()

                if batch_idx % 1 == 0:
                    print(
                        f"Client [{idx}/{args.num_users}] Epoch [{epoch}/{args.Gan_epochs}] Batch {batch_idx}/{len(train_loader)} \
                        Loss D: {loss_disc:.4f}, loss G: {loss_gen_total:.4f}")
            # save real image for each clients
            vutils.save_image(real, '%s/real_samples_for_client_%03d.png' % ('./results', idx), normalize=True)

        if epoch == args.Gan_epochs:
            noise_input = torch.randn(args.Batch_size, args.Channel_Noise, 1, 1)
            Generated_graph_set = []
            for k in range(args.Generator_paths):
                Generated_graph = Generator.paths[k](noise_input)
                Generated_graph_set.append(Generated_graph)
            Generated_graph_set = torch.cat(Generated_graph_set, dim=0)
            Augmentation_dataset = Data.TensorDataset(Generated_graph_set, torch.ones_like(Generated_graph_set))

        # for a simple visualization
        if args.Generator_paths == 1:
            noise_input = torch.randn(args.Batch_size, args.Channel_Noise, 1, 1)
            fake_1 = Generator.paths[0](noise_input).reshape(-1, args.Channels_Num, 64, 64)
            vutils.save_image(fake_1.data, '%s/fake_1_samples_epoch_%03d.png' % ('./results', epoch),normalize=True)
    return Augmentation_dataset


