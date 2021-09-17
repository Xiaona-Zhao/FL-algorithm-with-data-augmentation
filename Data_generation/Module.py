import torch.nn as nn
import torch
import torch.nn.functional as F

G_paths = 5


class DCGAN_Gen(nn.Module):
    def __init__(self, noise_dim, channels_num, feature_gen, G_paths):
        super(DCGAN_Gen, self).__init__()
        modules = nn.ModuleList()
        for _ in range(G_paths):
            modules.append(nn.Sequential(
                self._block(noise_dim,        feature_gen * 16, 4, 1, 0),
                self._block(feature_gen * 16, feature_gen * 8,  4, 2, 1),
                self._block(feature_gen * 8,  feature_gen * 4,  4, 2, 1),
                self._block(feature_gen * 4,  feature_gen * 2,  4, 2, 1),
                nn.ConvTranspose2d(feature_gen * 2, channels_num, kernel_size=4, stride=2, padding=1),
                nn.Tanh()))
            self.paths = modules

    def _block(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(output_channel), nn.ReLU())

    def forward(self, x):
        img = []
        for path in self.paths:
            img.append(path(x))
        img = torch.cat(img, dim=0)
        return img


class DCGAN_Gen_Single_Path(nn.Module):
    def __init__(self, noise_dim, channels_num, feature_gen, G_paths):
        super(DCGAN_Gen_Single_Path, self).__init__()
        self.seq = nn.Sequential(
            self._block(noise_dim, feature_gen * 16, 4, 1, 0),
            self._block(feature_gen * 16, feature_gen * 8, 4, 2, 1),
            self._block(feature_gen * 8, feature_gen * 4, 4, 2, 1),
            self._block(feature_gen * 4, feature_gen * 2, 4, 2, 1),
            nn.ConvTranspose2d(feature_gen * 2, channels_num, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(output_channel), nn.ReLU())

    def forward(self, x):
        output = self.seq(x)
        return output


class MPI_G(nn.Module):
    def __init__(self, noise_dim, G_paths):
        super(MPI_G, self).__init__()
        modules = nn.ModuleList()
        for _ in range(G_paths):
            modules.append(nn.Sequential(nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias = False),nn.BatchNorm2d(512), nn.ReLU(True),
                                  nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),nn.BatchNorm2d(256), nn.ReLU(True),
                                  nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),nn.BatchNorm2d(128), nn.ReLU(True),
                                  nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.ConvTranspose2d(64, 1, 4, 2, 1, bias = False),nn.Tanh()))
        self.paths = modules

    def forward(self, x):
        img = []
        for path in self.paths:
            img.append(path(x))
        img = torch.cat(img, dim=0)
        return img


class MPI_G_single_path(nn.Module):
    def __init__(self, noise_dim, channels_num, G_paths):
        super(MPI_G_single_path, self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
                                  nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
                                  nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
                                  nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.ConvTranspose2d(64, channels_num, 4, 2, 1, bias=False), nn.Tanh())
    def forward(self, input):
        output = self.main(input)
        return output



class MPI_D(nn.Module):
    def __init__(self, channels_num, G_paths):
        super(MPI_D, self).__init__()
        modules = nn.ModuleList()
        self.G_paths = G_paths
        self.main = nn.Sequential(nn.Conv2d(channels_num, 64, 4, 2, 1, bias = False),nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(64, 128, 4, 2, 1, bias = False),nn.BatchNorm2d(128),nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(128, 256, 4, 2, 1, bias = False),nn.BatchNorm2d(256),nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(256, 512, 4, 2, 1, bias = False),nn.BatchNorm2d(512),nn.LeakyReLU(0.2, inplace = True),)
        modules.append(nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias = False),nn.Sigmoid()))
        modules.append(nn.Sequential(nn.Conv2d(512, self.G_paths, 4, 1, 0, bias=False)))
        self.paths = modules

    def forward(self, input):
        x = self.main(input)
        output = self.paths[0](x)
        classifier = self.paths[1](x)
        return output, classifier


class FC_Generator(nn.Module):
    def __init__(self, z_dim, img_dim, G_paths):
        super().__init__()
        modules = nn.ModuleList()
        for _ in range(G_paths):
            modules.append(nn.Sequential(
                nn.Linear(z_dim, 256), nn.LeakyReLU(0.01),
                nn.Linear(256, img_dim), nn.Tanh()))
        self.paths = modules

    def forward(self, x):
        img = []
        for path in self.paths:
            img.append(path(x))
        img = torch.cat(img, dim=0)
        return img


class DCGAN_Disc(nn.Module):
    def __init__(self, channels_num, feature_disc, G_paths):
        super(DCGAN_Disc, self).__init__()
        modules = nn.ModuleList()
        self.G_paths = G_paths
        self.disc = nn.Sequential(
            nn.Conv2d(channels_num, feature_disc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(feature_disc * 1, feature_disc * 2, 4, 2, 1),
            self._block(feature_disc * 2, feature_disc * 4, 4, 2, 1),
            self._block(feature_disc * 4, feature_disc * 8, 4, 2, 1))
        modules.append(nn.Sequential(
            nn.Conv2d(feature_disc * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()))
        modules.append(nn.Sequential(
            nn.Conv2d(feature_disc * 8, self.G_paths, kernel_size=4, stride=2, padding=0)))
        self.paths = modules

    def _block(self, input_channels, output_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(output_channels), nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.disc(x)
        output = self.paths[0](x)
        classifier = F.log_softmax(self.paths[1](x), dim=1)
        return output, classifier


class FC_Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        modules = nn.ModuleList()
        self.disc = nn.Sequential(nn.Linear(in_features, 128), nn.LeakyReLU(0.01))
        modules.append(nn.Sequential(nn.Linear(128, 1), nn.Sigmoid()))
        modules.append(nn.Sequential(nn.Linear(128, 10)))
        self.paths = modules

    def forward(self, x):
        x = self.disc(x)
        output = self.paths[0](x)
        classifier = F.softmax(self.paths[1](x), dim=1)
        return output, classifier


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def demo():
    # net_G = FC_Generator(z_dim=64,img_dim=784)
    # net_D = FC_Discriminator(in_features=784)
    # net_G_single = DCGAN_Gen_Single_Path(noise_dim=100, channels_num=3, feature_gen=8, G_paths=1)

    net_MPI_G = MPI_G(noise_dim=100, G_paths=G_paths)
    net_MPI_D = MPI_D(channels_num=3, G_paths=G_paths)

    net_G = DCGAN_Gen(noise_dim=100, channels_num=3, feature_gen=8, G_paths=10)
    net_D = DCGAN_Disc(channels_num=3, feature_disc=8, G_paths=10)
    initialize_weights(net_G)
    initialize_weights(net_D)


    N, in_channels, H, W = 128, 3, 64, 64
    noise_dim = 100

    real = torch.randn((N, in_channels, H, W))  #(128, 3, 64,64)
    noise_input = torch.randn((N, noise_dim, 1, 1))  #(128, 100, 1, 1)

    disc, classifier = net_MPI_D(real)
    print(f'disc.shape = {disc.shape}')
    # print(f'disc = {disc}')
    print(f'classifier_shape = {classifier.shape}')
    # print(f'classifier = {classifier}')


    for k in range(G_paths):
        fake = net_MPI_G.paths[k](noise_input)
        print(f'fake.shape = {fake.shape} in G_path {k}')


    # for single path Generator:
    # fake = net_G_single(noise_input)
    # print(f'fake.shape = {fake.shape} in single path Generator')


# demo()
