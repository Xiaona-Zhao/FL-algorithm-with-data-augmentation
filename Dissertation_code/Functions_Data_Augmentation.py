import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Dissertation_code.Models import FC_Generator, FC_Discriminator, DCGAN_Generator, DCGAN_Discriminator, \
    DCGAN_Generator_Single_Path, WGAN_Generator, WGAN_Discriminator, initialize_weights, Discriminator_cifar10, \
    Generator_cifar10, Generator_cifar10_single_path, Discriminator_mnist, Generator_mnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


# print(f'device = {device}')


# get GAN model
def GAN_net(args, GAN_type):
    global net_G, net_D, net_G_single_path
    if GAN_type == 'DCGAN':
        net_G = DCGAN_Generator(noise_dim=args.Channel_Noise, channels_num=args.Channels_Num, feature_gen=8,
                                G_paths=args.Generator_paths)
        net_D = DCGAN_Discriminator(channels_num=args.Channels_Num, feature_disc=8, G_paths=args.Generator_paths)
        initialize_weights(net_G)
        initialize_weights(net_D)
    elif GAN_type == 'WGAN':
        net_G = WGAN_Generator(noise_dim=args.Channel_Noise, channels_num=args.Channels_Num, feature_gen=8,
                               G_paths=args.Generator_paths)
        net_D = WGAN_Discriminator(channels_num=args.Channels_Num, feature_disc=8, G_paths=args.Generator_paths)
        initialize_weights(net_G)
        initialize_weights(net_D)
    elif GAN_type == 'cifar10_GAN':
        net_G = Generator_cifar10(noise_dim=args.Channel_Noise, G_paths=args.Generator_paths)
        net_D = Discriminator_cifar10(channels_num=args.Channels_Num, G_paths=args.Generator_paths)
    elif GAN_type == 'mnist_GAN':
        net_G = Generator_mnist(noise_dim=args.Channel_Noise, G_paths=args.Generator_paths)
        net_D = Discriminator_mnist(channels_num=args.Channels_Num, G_paths=args.Generator_paths)
    elif GAN_type == 'FCGAN':
        net_G = FC_Generator(z_dim=100, img_dim=64, G_paths=4)
        net_D = FC_Discriminator(in_features=8)
    else:
        print('GAN_type undefined')

    Generator = net_G.to(device)
    Discriminator_list = []
    for i in range(args.num_discriminator):
        Discriminator = net_D.to(device)
        Discriminator_list.append(Discriminator)
    return Generator, Discriminator_list


# get GAN optimizer
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
            opt_gen.append(optim.Adagrad(Generator.paths[i].parameters(), lr=0.008, eps=1e-08))
        for i in range(args.num_discriminator):
            Discriminator = Discriminator_list[i]
            opt_disc = optim.Adagrad(Discriminator.parameters(), lr=0.001, eps=1e-08)
            opt_disc_list.append(opt_disc)
    return opt_gen, opt_disc_list


# def loss function
criterion = nn.BCELoss().to(device)


# Training process for discriminator
def Disc_train(args, Generator, Discriminator, real, noise, loss_disc_total):
    Discriminator_real, classifier_r = Discriminator(real)
    Discriminator_real = Discriminator_real.reshape(-1)
    loss_disc_real = criterion(Discriminator_real, torch.ones_like(Discriminator_real))

    loss_classifier_total = 0
    loss_disc_fake_total = 0
    for i in range(args.Generator_paths):
        fake = Generator.paths[i](noise)
        Discriminator_fake, classifier_f = Discriminator(fake.detach())
        Discriminator_fake = Discriminator_fake.reshape(-1)

        classifier_f = classifier_f.squeeze(-1)

        loss_disc_fake = criterion(Discriminator_fake, torch.zeros_like(Discriminator_fake))
        loss_disc_fake_total += loss_disc_fake

        target = Variable(Tensor(real.size(0)).fill_(i), requires_grad=False)
        target = target.type(Tensor)
        target = target.unsqueeze(1)

        loss_classifier = F.nll_loss(classifier_f, target) * args.classifier_parameter
        loss_classifier_total += loss_classifier

    loss_classifier_avg = loss_classifier_total / args.Generator_paths
    loss_disc_fake_avg = loss_disc_fake_total / args.Generator_paths

    loss_disc = (loss_disc_fake_avg + loss_disc_real) / 2
    loss_disc_total += loss_disc + loss_classifier_avg
    return loss_disc_total


# Train process for Generator
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


def GAN_loader(dataset, idxs):
    idxs_train = idxs[:int(1 * len(idxs))]
    train_loader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=50, shuffle=True)
    return train_loader

