import torch.nn as nn
import torch
import torch.nn.functional as F


class DCGAN_Generator(nn.Module):
    def __init__(self, noise_dim, channels_num, feature_gen, G_paths):
        super(DCGAN_Generator, self).__init__()
        modules = nn.ModuleList()
        for _ in range(G_paths):
            modules.append(nn.Sequential(
                self._block(noise_dim, feature_gen * 16, 4, 1, 0),
                self._block(feature_gen * 16, feature_gen * 8, 4, 2, 1),
                self._block(feature_gen * 8, feature_gen * 4, 4, 2, 1),
                self._block(feature_gen * 4, feature_gen * 2, 4, 2, 1),
                nn.ConvTranspose2d(feature_gen * 2, channels_num, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
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


class DCGAN_Generator_Single_Path(nn.Module):
    def __init__(self, noise_dim, channels_num, feature_gen, G_paths):
        super(DCGAN_Generator_Single_Path, self).__init__()
        self.seq = nn.Sequential(
            self._block(noise_dim, feature_gen * 16, 4, 1, 0),
            self._block(feature_gen * 16, feature_gen * 8, 4, 2, 1),
            self._block(feature_gen * 8, feature_gen * 4, 4, 2, 1),
            self._block(feature_gen * 4, feature_gen * 2, 4, 2, 1),
            nn.ConvTranspose2d(feature_gen * 2, channels_num, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def _block(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(output_channel), nn.ReLU())

    def forward(self, x):
        output = self.seq(x)
        return output


class DCGAN_Discriminator(nn.Module):
    def __init__(self, channels_num, feature_disc, G_paths):
        super(DCGAN_Discriminator, self).__init__()
        modules = nn.ModuleList()
        self.G_paths = G_paths
        self.disc = nn.Sequential(
            nn.Conv2d(channels_num, feature_disc, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2),
            self._block(feature_disc * 1, feature_disc * 2, 4, 2, 1),
            self._block(feature_disc * 2, feature_disc * 4, 4, 2, 1),
            self._block(feature_disc * 4, feature_disc * 8, 4, 2, 1))
        modules.append(nn.Sequential(
            nn.Conv2d(feature_disc * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=0),
            nn.Sigmoid()))
        modules.append(nn.Sequential(
            nn.Conv2d(feature_disc * 8, self.G_paths, kernel_size=(4, 4), stride=(2, 2), padding=0)))
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


class WGAN_Generator(nn.Module):
    def __init__(self, noise_dim, channels_num, feature_gen, G_paths):
        super(WGAN_Generator, self).__init__()
        modules = nn.ModuleList()
        for _ in range(G_paths):
            modules.append(nn.Sequential(
                self._block(in_channels=noise_dim, out_channels=feature_gen * 16, kernel_size=4, stride=1, padding=0),
                self._block(in_channels=feature_gen * 16, out_channels=feature_gen * 8, kernel_size=4, stride=2,
                            padding=1),
                self._block(in_channels=feature_gen * 8, out_channels=feature_gen * 4, kernel_size=4, stride=2,
                            padding=1),
                self._block(in_channels=feature_gen * 4, out_channels=feature_gen * 2, kernel_size=4, stride=2,
                            padding=1),
                nn.ConvTranspose2d(feature_gen * 2, channels_num, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                nn.Tanh()))
        self.paths = modules

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                             nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        img = []
        for path in self.paths:
            img.append(path(x))
        img = torch.cat(img, dim=0)
        return img


class WGAN_Discriminator(nn.Module):
    def __init__(self, channels_num, feature_disc, G_paths):
        super(WGAN_Discriminator, self).__init__()
        modules = nn.ModuleList()
        self.G_paths = G_paths
        self.main = nn.Sequential(nn.Conv2d(channels_num, feature_disc, kernel_size=(4, 4), stride=(2, 2), padding=1),
                                  nn.LeakyReLU(0.2),
                                  self._block(in_channels=feature_disc * 1, out_channels=feature_disc * 2,
                                              kernel_size=4, stride=2, padding=1),
                                  self._block(in_channels=feature_disc * 2, out_channels=feature_disc * 4,
                                              kernel_size=4, stride=2, padding=1),
                                  self._block(in_channels=feature_disc * 4, out_channels=feature_disc * 8,
                                              kernel_size=4, stride=2, padding=1))
        modules.append(nn.Conv2d(feature_disc * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=0)),
        modules.append(nn.Conv2d(feature_disc * 8, self.G_paths, kernel_size=(4, 4), stride=(2, 2), padding=0))
        self.paths = modules

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, ),
            nn.InstanceNorm2d(out_channels, affine=True), nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.main(x)
        output = self.paths[0](x)
        classifier = self.paths[1](x)
        return output, classifier


class Generator_cifar10(nn.Module):
    def __init__(self, noise_dim, G_paths):
        super(Generator_cifar10, self).__init__()
        modules = nn.ModuleList()
        for _ in range(G_paths):
            modules.append(nn.Sequential(nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256),
                                         nn.ReLU(True),
                                         nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128),
                                         nn.ReLU(True),
                                         nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64),
                                         nn.ReLU(True),
                                         nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh()))
        self.paths = modules

    def forward(self, x):
        img = []
        for path in self.paths:
            img.append(path(x))
        img = torch.cat(img, dim=0)
        return img


class Discriminator_cifar10(nn.Module):
    def __init__(self, channels_num, G_paths):
        super(Discriminator_cifar10, self).__init__()
        modules = nn.ModuleList()
        self.G_paths = G_paths
        self.main = nn.Sequential(nn.Conv2d(channels_num, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False), nn.Sigmoid()))
        modules.append(nn.Sequential(nn.Conv2d(512, self.G_paths, 4, 1, 0, bias=False)))
        self.paths = modules

    def forward(self, input):
        x = self.main(input)
        output = self.paths[0](x)
        classifier = self.paths[1](x)
        return output, classifier


class Generator_cifar10_single_path(nn.Module):
    def __init__(self, noise_dim, G_paths):
        super(Generator_cifar10_single_path, self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
                                  nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
                                  nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        output = self.main(input)
        return output


class Generator_mnist(nn.Module):
    def __init__(self, noise_dim, G_paths):
        super(Generator_mnist, self).__init__()
        modules = nn.ModuleList()
        for _ in range(G_paths):
            modules.append(nn.Sequential(nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512),
                                         nn.ReLU(True),
                                         nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256),
                                         nn.ReLU(True),
                                         nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128),
                                         nn.ReLU(True),
                                         nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64),
                                         nn.ReLU(True),
                                         nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), nn.Tanh()))
        self.paths = modules

    def forward(self, x):
        img = []
        for path in self.paths:
            img.append(path(x))
        img = torch.cat(img, dim=0)
        return img


class Discriminator_mnist(nn.Module):
    def __init__(self, channels_num, G_paths):
        super(Discriminator_mnist, self).__init__()
        modules = nn.ModuleList()
        self.G_paths = G_paths
        self.main = nn.Sequential(nn.Conv2d(channels_num, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False), nn.Sigmoid()))
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
            modules.append(nn.Sequential(nn.Linear(z_dim, 256), nn.LeakyReLU(0.01), nn.Linear(256, img_dim), nn.Tanh()))
        self.paths = modules

    def forward(self, x):
        img = []
        for path in self.paths:
            img.append(path(x))
        img = torch.cat(img, dim=0)
        return img


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


class MNIST_classification_model(nn.Module):
    def __init__(self):
        super(MNIST_classification_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=(1, 1)), nn.BatchNorm2d(n1x1), nn.ReLU(True))

        self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=(1, 1)), nn.BatchNorm2d(n3x3red),
                                nn.ReLU(True),
                                nn.Conv2d(n3x3red, n3x3, kernel_size=(3, 3), padding=1),
                                nn.BatchNorm2d(n3x3),
                                nn.ReLU(True),
                                )

        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=(1, 1)),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=(1, 1)),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


# GoogLeNet
class CIFAR10_classification_model(nn.Module):
    def __init__(self):
        super(CIFAR10_classification_model, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(82944, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def demo():
    G_paths = 4
    input_dim = 3
    noise_dim = 100
    N, in_channels, H, W = 128, 3, 64, 64

    Generator = Generator_cifar10(noise_dim=noise_dim, G_paths=G_paths)
    Discriminator = Discriminator_cifar10(channels_num=input_dim, G_paths=G_paths)

    initialize_weights(Generator)
    initialize_weights(Discriminator)

    real = torch.randn((N, in_channels, H, W))  # (128, 3, 64, 64)
    noise_input = torch.randn((N, noise_dim, 1, 1))  # (128, 100, 1, 1)

    disc, classifier = Discriminator(real)
    print(f'disc.shape = {disc.shape}')
    print(f'classifier_shape = {classifier.shape}')

    for k in range(G_paths):
        fake = Generator.paths[k](noise_input)
        print(f'fake.shape = {fake.shape} in G_path {k}')

# demo()
