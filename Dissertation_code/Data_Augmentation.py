import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from Dissertation_code.Functions_Data_Augmentation import GAN_net, GAN_optimizer, Disc_train, Gen_train, GAN_loader
from Dissertation_code.Models import MNIST_classification_model, CIFAR10_classification_model
from Dissertation_code.MNIST_classification import MNIST_classifier
from Dissertation_code.Cifar_classification import Cifar10_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def Get_generator_model(args, train_dataset, user_groups, Generator, Discriminator_list, opt_gen, opt_disc_list):
    print('Generator training triggered.')
    global Generator_model
    Generator.train()

    for Discriminator in Discriminator_list:
        Discriminator.train()
    for epoch in range(args.Gan_epochs):
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # GAN training in distributed framework
        for idx in idxs_users:
            assert args.num_users == args.num_discriminator
            train_loader = GAN_loader(train_dataset, list(user_groups[idx]))

            for batch_idx, (real, _) in enumerate(train_loader):
                real = real.to(device)
                noise = torch.randn(args.Training_Batch_size, args.Channel_Noise, 1, 1).to(device)

                loss_gen_total = 0
                loss_disc_total = 0

                # update discriminator
                loss_disc = Disc_train(args, Generator=Generator, Discriminator=Discriminator_list[idx], real=real,
                                       noise=noise, loss_disc_total=loss_disc_total)
                Discriminator_list[idx].zero_grad()
                loss_disc.backward(retain_graph=True)
                opt_disc_list[idx].step()

                # update multi_path generator
                for path in range(args.Generator_paths):
                    loss_gen_total = Gen_train(args, Path=path, Generator=Generator,
                                               Discriminator=Discriminator_list[idx],
                                               real=real, noise=noise, loss_gen_total=loss_gen_total)
                    Generator.paths[path].zero_grad()
                    loss_gen_total.backward()
                    opt_gen[path].step()

        # visualization of generated image
        if args.visualization is True and args.Generator_paths == 4:
            Generator = Generator.to(device)
            noise_test = torch.randn(args.Testing_Batch_size, args.Channel_Noise, 1, 1).to(device)
            fake_1 = Generator.paths[0](noise_test).reshape(-1, args.Channels_Num, 64, 64)
            fake_2 = Generator.paths[1](noise_test).reshape(-1, args.Channels_Num, 64, 64)
            fake_3 = Generator.paths[2](noise_test).reshape(-1, args.Channels_Num, 64, 64)
            fake_4 = Generator.paths[3](noise_test).reshape(-1, args.Channels_Num, 64, 64)

            vutils.save_image(fake_1.data, '%s/fake_1_samples_epoch_%03d.png' % ('./results/Generated_image', epoch),
                              normalize=True)
            vutils.save_image(fake_2.data, '%s/fake_2_samples_epoch_%03d.png' % ('./results/Generated_image', epoch),
                              normalize=True)
            vutils.save_image(fake_3.data, '%s/fake_3_samples_epoch_%03d.png' % ('./results/Generated_image', epoch),
                              normalize=True)
            vutils.save_image(fake_4.data, '%s/fake_4_samples_epoch_%03d.png' % ('./results/Generated_image', epoch),
                              normalize=True)

        # save generator model
        if epoch == args.Gan_epochs - 1:
            torch.save(Generator.state_dict(),
                       '%s/Generator_model_epoch_%03d.pth' % ('./results/Generator_model', epoch))
            Generator_model = Generator()
    print('Generator training finished.')
    return Generator_model


def Get_classifier_model(args):
    global classifier_model
    if args.dataset == 'mnist':
        basic_model = MNIST_classification_model().to(device)
        classifier_model = MNIST_classifier(model=basic_model, num_epochs=20)
    if args.dataset == 'cifar10':
        basic_model = CIFAR10_classification_model().to(device)
        classifier_model = Cifar10_classifier(model=basic_model, num_epochs=200)
    torch.save(classifier_model.state_dict(),
               '%s/Generator_model.pth' % './results/Classifier_model')
    return classifier_model


def Get_augmentation_dataset(args, Generator, Classifier, Augmentation_size):
    print('Augmentation dataset generation triggered.')

    Generated_image_set = []
    Generator_model = Generator()
    classifier_model = Classifier()

    # get generated images for data augmentation
    Generator_model.eval()
    noise_input = torch.randn(Augmentation_size, 128, 1, 1)
    for k in range(args.Generator_paths):
        Generated_image = Generator_model.paths[k](noise_input)
        Generated_image_set.append(Generated_image)
    Generated_image_set = torch.cat(Generated_image_set, dim=0)

    # label the generated images for augmentation dataset
    classifier_model.eval()
    prediction = classifier_model(Generated_image_set)
    _, label = prediction.max(1)
    label = torch.tensor(label, dtype=torch.int64)

    # Generate augmentation dataset
    Augmentation_dataset = Data.TensorDataset(Generated_image_set, label)
    print('Augmentation dataset generation finished.')
    return Augmentation_dataset
