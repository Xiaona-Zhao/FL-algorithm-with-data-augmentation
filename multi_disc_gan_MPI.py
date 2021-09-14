
# MD-GAN	: Generative Adversarial Network, Multi-Discriminator GenerativeAdversarial Networks for Distributed Datasets
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

batchsize = 100
imagesize = 64
LOSS_GEN = []
LOSS_DISC = []

transform = transforms.Compose([transforms.Resize(imagesize), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
device = "cuda" if torch.cuda.is_available() else "cpu"
# nc = 3
dataset = dset.CIFAR10(root = './data/CIFAR10', download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True, num_workers = 2)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),nn.BatchNorm2d(512), nn.ReLU(True),
                                  nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),nn.BatchNorm2d(256), nn.ReLU(True),
                                  nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),nn.BatchNorm2d(128), nn.ReLU(True),
                                  nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),nn.Tanh())

    def forward(self, input):
        output = self.main(input)
        return output


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias = False),nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(64, 128, 4, 2, 1, bias = False),nn.BatchNorm2d(128),nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(128, 256, 4, 2, 1, bias = False),nn.BatchNorm2d(256),nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(256, 512, 4, 2, 1, bias = False),nn.BatchNorm2d(512),nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(512, 1, 4, 1, 0, bias = False),nn.Sigmoid())

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netG = G()
netG.apply(weights_init).to(device)
netD = D()
netD.apply(weights_init).to(device)


criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


def shuffleDiscriminators():
    if(rank !=0 ):
        layer_num = 0
        for param in netD.parameters():
            outdata = param.data.numpy().copy()
            indata = None


            if (rank != size - 1):
                comm.send(outdata, dest = rank+1, tag = 1)
            if(rank != 1):
                indata = comm.recv(source = rank - 1, tag = 1)
            if (rank == size - 1):
                comm.send(outdata, dest = 1, tag = 2)
            if(rank == 1):
                indata = comm.recv(source = size - 1, tag = 2)
            param.data = torch.from_numpy(indata)
            layer_num += 1

num_epoch = 30
for epoch in range(num_epoch):
    if (epoch % 2 == 0):
        shuffleDiscriminators()


    for i, data in enumerate(dataloader):
        netD.zero_grad()
        real, _ = data

        # 1.
        # output取自input， input取自real，error 在output和target之间
        input = Variable(real).to(device)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)

        # output取自fake，fake取自noise，error在output和target之间
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)).to(device)
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)

        # 将两个error相加，向后传播
        errD = errD_fake + errD_real
        errD.backward()
        optimizerD.step()

        # 2.
        # output取自之前相同的fake，error在output和target之间
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()


        print('[%d / %d][%d / %d] Loss_D: %.4f Loss_G: %.4f' % (epoch, num_epoch, i, len(dataloader), errD.item(), errG.item()))
        if i % 100==0:
            vutils.save_image(real, '%s/real_samples.png' % './results', normalize=True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ('./results', epoch), normalize=True)
            LOSS_GEN.append(errG.item())
            LOSS_DISC.append(errD.item())

print(LOSS_GEN)
print(LOSS_DISC)





# Defining the copy of the generator to shuffle between diffrent severs
# def copyGenerator():
#     layer_num = 0
#     for param in netG.parameters():
#         #print(rank, "started")
#         if (rank == 0):
#             data = param.data.numpy().copy()
#             #print(rank, data.shape)
#         else:
#             data = None
#             #print(rank, data.shape)
#
#         #print(rank, "before bcast")
#         #comm.Barrier()
#         data = comm.bcast(data, root = 0)
#         #print(rank, "after bcast")
#         if (rank != 0):
#             param.data = torch.from_numpy(data)
#             #print("Node rank " + str(rank) + " has synched generator layer " + str(layer_num))
#
#         layer_num += 1
#         #comm.Barrier()
