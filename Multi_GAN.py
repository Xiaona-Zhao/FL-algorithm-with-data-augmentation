import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from Data_generation.Module import FC_Generator, FC_Discriminator, DCGAN_Gen, DCGAN_Disc, DCGAN_Gen_Single_Path, \
    initialize_weights
from Data_generation.Create_dataset import FL_dataset

data_root = '../dataset/CIFAR10'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# some hyper_parameters
data_name = 'cifar10'
Learning_rate = 2e-4


Channels_Num = 3
Channel_Noise = 100
Batch_size = 100
frac = 1
non_iid_alpha = 0.01

Image_size = 64
Num_epochs = 101
optimizer_type = 'Adam'

num_users = 10
Generator_paths = 10
num_discriminator = 10
classifier_parameter = 1


# 定义transform模块
def Data_transform(data_name):
    global Transform
    if data_name == 'cifar10':
        Transform = transforms.Compose([transforms.Resize(Image_size), transforms.ToTensor(),
                                        transforms.Normalize([0.5 for _ in range(Channels_Num)],
                                                             [0.5 for _ in range(Channels_Num)])])
    elif data_name == 'mnist':
        Transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return Transform


# 调用transform
transform = Data_transform(data_name=data_name)


# Datasplit from FedAvg source code
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
        # return image.clone().detach(), label.clone().detach()


# get_dataset model, user_group will only work when data_type == 'FL_mode'
def get_dataset(data_name, data_type):
    global train_dataset, test_dataset, train_loader, test_loader, user_groups
    if data_type == 'origin':
        if data_name == 'mnist':
            train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=False)
            train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=True)
            user_groups = None
        elif data_name == 'cifar10':
            train_dataset = datasets.CIFAR10(root=data_root, train=True, transform=transform, download=True)
            test_dataset = datasets.CIFAR10(root=data_root, train=False, transform=transform, download=False)
            train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=True)
            user_groups = None
    elif data_type == 'FL_mode':
        train_dataset, test_dataset, user_groups = FL_dataset(world_size=num_users, dataset=data_name,
                                                              non_iid_alpha=non_iid_alpha)
        train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=True)
    else:
        print('Data type has not been defined')
    return train_dataset, test_dataset, train_loader, test_loader, user_groups


# 调用get_dataset
train_dataset, test_dataset, train_loader, test_loader, user_groups = get_dataset(data_name='cifar10', data_type='FL_mode')
# train_dataset, test_dataset, train_loader, test_loader, user_groups = get_dataset(data_name='cifar10',
#                                                                                   data_type='origin')


# 定义, 初始化GAN网络, One Generator with multi paths, and multi Discriminator corresponding to each FL client dataset
def net(GAN_type, Generator_paths, num_discriminator):
    global net_G, net_D, net_G_single_path
    if GAN_type == 'DCGAN':
        net_G = DCGAN_Gen(noise_dim=Channel_Noise, channels_num=Channels_Num, feature_gen=8, G_paths=Generator_paths)
        net_G_single_path = DCGAN_Gen(noise_dim=Channel_Noise, channels_num=Channels_Num, feature_gen=8, G_paths=1)
        net_D = DCGAN_Disc(channels_num=Channels_Num, feature_disc=8)
        initialize_weights(net_G)
        initialize_weights(net_G_single_path)
        initialize_weights(net_D)
    else:
        print('GAN_type undefined')

    # sent neuron network to device
    Generator = net_G.to(device)
    Generator_single_path = net_G_single_path.to(device)
    Discriminator_list = []
    for i in range(num_discriminator):
        Discriminator = net_D.to(device)
        Discriminator_list.append(Discriminator)
    return Generator, Generator_single_path, Discriminator_list


Generator, Generator_single_path, Discriminator_list = net(GAN_type='DCGAN', Generator_paths=Generator_paths,
                                                           num_discriminator=num_discriminator)


def optimizer(Generator, Discriminator_list, Generator_paths, num_discriminator, optimizer_type):
    opt_gen = []
    opt_disc_list = []
    if optimizer_type == 'Adam':
        for i in range(Generator_paths):
            opt_gen.append(optim.Adam(Generator.paths[i].parameters(), lr=Learning_rate, betas=(0.5, 0.999)))
        for i in range(num_discriminator):
            Discriminator = Discriminator_list[i]
            opt_disc = optim.Adam(Discriminator.parameters(), lr=Learning_rate, betas=(0.5, 0.999))
            opt_disc_list.append(opt_disc)

    elif optimizer_type == 'Adagrad':
        for i in range(Generator_paths):
            opt_gen.append(optim.Adagrad(Generator.paths[i].parameters(), lr=Learning_rate,))
        for i in range(num_discriminator):
            Discriminator = Discriminator_list[i]
            opt_disc = optim.Adagrad(Discriminator.parameters(), lr=Learning_rate,)
            opt_disc_list.append(opt_disc)
    return opt_gen, opt_disc_list


opt_gen, opt_disc_list = optimizer(Generator=Generator, Discriminator_list=Discriminator_list,
                                   Generator_paths=Generator_paths, num_discriminator=num_discriminator,
                                   optimizer_type=optimizer_type)

# 定义损失函数
criterion = nn.BCELoss().to(device)
# criterion = nn.CrossEntropyLoss()

# 写个logger
writer_real = SummaryWriter(f'../MultiGAN federated learning/logs/real')
writer_fake = SummaryWriter(f'../MultiGAN federated learning/logs/fake')
step = 0


def Disc_train(Generator, Discriminator, real, noise, loss_disc_total):
    # 计算输入为real的DISC_loss
    Discriminator_real, classifier_r = Discriminator(real)
    Discriminator_real = Discriminator_real.reshape(-1)  # Discriminator_real.shape = [128]
    loss_disc_real = criterion(Discriminator_real, torch.ones_like(Discriminator_real))

    loss_classifier_total = 0
    loss_disc_fake_total = 0
    for i in range(Generator_paths):
        # 计算输入为fake的DISC_loss
        fake = Generator.paths[i](noise)  # fake.shape = [128, 3, 64, 64]
        Discriminator_fake, classifier_f = Discriminator(fake.detach())

        Discriminator_fake = Discriminator_fake.reshape(-1)  # Discriminator_fake.shape = [128]
        classifier_f = classifier_f.squeeze(-1)

        loss_disc_fake = criterion(Discriminator_fake, torch.zeros_like(Discriminator_fake))
        loss_disc_fake_total += loss_disc_fake

        # 将 target进行维度重构
        target = Variable(Tensor(real.size(0)).fill_(i), requires_grad=False)  # target.shape = [128]
        target = target.type(Tensor)
        target = target.unsqueeze(1)

        loss_classifier = F.nll_loss(classifier_f, target) * classifier_parameter
        loss_classifier_total += loss_classifier

    loss_classifier_avg = loss_classifier_total / Generator_paths
    loss_disc_fake_avg = loss_disc_fake_total / Generator_paths

    # 将两个loss加和/2为DISC的loss
    loss_disc = (loss_disc_fake_avg + loss_disc_real) / 2
    loss_disc_total += loss_disc + loss_classifier_avg
    return loss_disc_total


def Gen_train(Path, Generator, Discriminator, real, noise, loss_gen_total):
    fake = Generator.paths[Path](noise)
    output, classifier = Discriminator(fake)
    # classifier = classifier.view(-1)
    # classifier = classifier.squeeze(-1)

    loss_gen = criterion(output, torch.ones_like(output))

    # target = Variable(Tensor(real.size(0)).fill_(Path), requires_grad=False)
    # target = target.type(Tensor)
    # target = target.unsqueeze(1)
    # loss_classifier = F.nll_loss(classifier, target) * classifier_parameter

    loss_gen_total = loss_gen  # + loss_classifier
    return loss_gen_total


def client_train_loader(idx, user_groups):
    idxs = user_groups[idx]  # len(idxs) = 5000
    idxs_train = idxs[:int(len(idxs))]  # idxs_train = [42597, 42618, 42621, 42622, 42628, 42630, 42644, 42649...]
    train_loader = DataLoader(DatasetSplit(train_dataset, idxs_train), batch_size=Batch_size, shuffle=True)
    return train_loader





def training():
    Generator.train()
    Generator_single_path.train()
    for Discriminator in Discriminator_list:
        Discriminator.train()

    for epoch in range(Num_epochs):
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)  # idxs_users = [9 8 1 3 0 4 5 2 6 7]
        for idx in idxs_users:
            assert len(idxs_users) == num_discriminator

            # define train loader for each client, and training in coresponding Discriminator
            train_loader = client_train_loader(idx=idx, user_groups=user_groups)

            for batch_idx, (real, _) in enumerate(train_loader):
                # print(f'batch number = {batch_idx} in range {len(train_loader)}')
                real = real.to(device)
                noise = torch.randn(Batch_size, Channel_Noise, 1, 1).to(device)

                loss_gen_total = 0
                loss_disc_total = 0

                # training Discriminator
                loss_disc = Disc_train(Generator=Generator, Discriminator=Discriminator_list[idx],
                                       real=real, noise=noise, loss_disc_total=loss_disc_total)
                Discriminator_list[idx].zero_grad()
                loss_disc.backward(retain_graph=True)
                opt_disc_list[idx].step()

                # training Generator
                for path in range(Generator_paths):
                    loss_gen_total = Gen_train(Path=path, Generator=Generator, Discriminator=Discriminator_list[idx],
                                               real=real, noise=noise, loss_gen_total=loss_gen_total)
                    Generator.paths[path].zero_grad()
                    loss_gen_total.backward()
                    opt_gen[path].step()

                if batch_idx % 10 == 0:
                    print(
                        f"Client [{idx}/{len(idxs_users)}] Epoch [{epoch}/{Num_epochs}] Batch {batch_idx}/{len(train_loader)} \
                        Loss D: {loss_disc:.4f}, loss G: {loss_gen_total:.4f}")
                    vutils.save_image(real, '%s/real_samples.png' % './results', normalize=True)
        if epoch % 10 == 0:
            noise_test = torch.randn(Batch_size, Channel_Noise, 1, 1).to(device)
            fake_1 = Generator.paths[0](noise_test).reshape(-1, 3, 64, 64)
            fake_3 = Generator.paths[2](noise_test).reshape(-1, 3, 64, 64)
            fake_5 = Generator.paths[4](noise_test).reshape(-1, 3, 64, 64)
            fake_7 = Generator.paths[6](noise_test).reshape(-1, 3, 64, 64)

            vutils.save_image(fake_1.data, '%s/fake_1_samples_epoch_%03d.png' % ('./results', epoch),
                              normalize=True)
            vutils.save_image(fake_3.data, '%s/fake_3_samples_epoch_%03d.png' % ('./results', epoch),
                                normalize=True)
            vutils.save_image(fake_5.data, '%s/fake_5_samples_epoch_%03d.png' % ('./results', epoch),
                              normalize=True)
            vutils.save_image(fake_7.data, '%s/fake_7_samples_epoch_%03d.png' % ('./results', epoch),
                              normalize=True)


training()


def single_generator_training():
    # train_dataset, test_dataset, train_loader, test_loader, user_groups = get_dataset(data_name='cifar10',
    #                                                                                   data_type='origin')

    opt_generator = optim.Adam(Generator_single_path.parameters(), lr=Learning_rate, betas=(0.5, 0.999))
    opt_discriminator = optim.Adam(Discriminator_list[0].parameters(), lr=Learning_rate, betas=(0.5, 0.999))

    Generator_single_path.train()  # using the generator with single path, as well as the original one
    Discriminator_list[0].train()  # using the first discriminator from the Discriminator_list

    for epoch in range(Num_epochs):
        for batch_idx, (real, _) in enumerate(train_loader):
            # print(f'batch number = {batch_idx} in range {len(train_loader)}')
            real = real.to(device)
            noise = torch.randn(Batch_size, Channel_Noise, 1, 1).to(device)
            fake = Generator_single_path(noise)

            # train Disc - loss function
            Discriminator_real, _ = Discriminator_list[0](real)
            Discriminator_real = Discriminator_real.reshape(-1)  # Discriminator_real.shape = [128]
            loss_disc_real = criterion(Discriminator_real, torch.ones_like(Discriminator_real))

            Discriminator_fake, _ = Discriminator_list[0](fake.detach())
            Discriminator_fake = Discriminator_fake.reshape(-1)
            loss_disc_fake = criterion(Discriminator_fake, torch.zeros_like(Discriminator_fake))

            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            # train Disc backward
            Discriminator_list[0].zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_discriminator.step()

            # train Gen - loss function
            output, _ = Discriminator_list[0](fake)
            loss_gen = criterion(output, torch.ones_like(output))

            # train Gen - backward
            Generator_single_path.zero_grad()
            loss_gen.backward()
            opt_generator.step()

            # show loss value of Gen and Disc
            if batch_idx % 100 == 0: #for batch_size = 100, there will be total 500 bath_idx
                print(
                    f"Epoch [{epoch}/{Num_epochs}] Batch {batch_idx}/{len(train_loader)} \
                        Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
                vutils.save_image(real, '%s/real_samples.png' % './results', normalize=True)

        # visualize the generated graph every 5 epoch
        if epoch % 5 == 0:
            noise_test = torch.randn(Batch_size, Channel_Noise, 1, 1)
            fake = Generator_single_path(noise_test).reshape(-1, 3, 64, 64)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ('./results', epoch),
                              normalize=True)
