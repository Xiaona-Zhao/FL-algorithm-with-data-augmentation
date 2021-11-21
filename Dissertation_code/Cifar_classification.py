import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from Dissertation_code.Models import CIFAR10_classification_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10_classification_model().to(device)

# get training dataset
transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
train_dataset = datasets.CIFAR10(root="../dataset/cifar10", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root="../dataset/cifar10", train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

# get loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device=device)
            label = label.to(device=device)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def Cifar10_classifier(model, num_epochs):
    print('Classifier training triggered')
    for epoch in range(num_epochs):
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device=device)
            label = label.to(device=device)

            scores = model(data)
            loss = criterion(scores, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    Cifar10_classifier_model = model
    accuracy = check_accuracy(test_loader, Cifar10_classifier_model)
    print(f"Accuracy of MNIST_classifier_model on test set: {accuracy * 100:.2f}")
    print('Classifier training finished')
    return Cifar10_classifier_model


def demo():
    Cifar10_classifier(model=model, num_epochs=2)
    return

# demo()
