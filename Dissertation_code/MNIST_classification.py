import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from Dissertation_code.Models import MNIST_classification_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
train_dataset = datasets.MNIST(root="../dataset/mnist", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="../dataset/mnist", train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# Initialize network
model = MNIST_classification_model().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


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


def MNIST_classifier(model, num_epochs):
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
    MNIST_classifier_model = model
    accuracy = check_accuracy(test_loader, MNIST_classifier_model)
    print(f"Accuracy of MNIST_classifier_model on test set: {accuracy * 100:.2f}")
    print('Classifier training finished')
    return MNIST_classifier_model


def demo():
    MNIST_classifier(model=model, num_epochs=2)
    return

# demo()
