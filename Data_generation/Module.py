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
    net_G = DCGAN_Gen(noise_dim=100, channels_num=3, feature_gen=8, G_paths=10)
    net_D = DCGAN_Disc(channels_num=3, feature_disc=8, G_paths=10)
    initialize_weights(net_G)
    initialize_weights(net_D)

    print(net_G)
    print(net_D)

    N, in_channels, H, W = 128, 3, 64, 64
    noise_dim = 100

    real = torch.randn((N, in_channels, H, W))  #(128, 3, 64,64)
    noise_input = torch.randn((N, noise_dim, 1, 1))  #(128, 100, 1, 1)

    disc, classifier = net_D(real)
    print(f'disc.shape = {disc.shape}')
    print(f'classifier_shape = {classifier.shape}')

    for k in range(G_paths):
        fake = net_G.paths[k](noise_input)
        print(f'fake.shape = {fake.shape} in G_path {k}')


    # for single path Generator:
    # fake = net_G_single(noise_input)
    # print(f'fake.shape = {fake.shape} in single path Generator')


# demo()
