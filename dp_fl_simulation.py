import torch
torch.classes.__path__ = []

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
import random

# Simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Add Gaussian or Laplace noise to model gradients
def add_dp_noise(model, scale, mechanism="Gaussian"):
    for param in model.parameters():
        if param.requires_grad:
            if mechanism == "Gaussian":
                noise = torch.normal(0, scale, size=param.grad.shape).to(param.device)
            else:  # Laplace
                noise = torch.distributions.Laplace(0, scale).sample(param.grad.shape).to(param.device)
            param.grad += noise

# Federated Learning Simulation
def run_dp_federated_learning(epsilon, clip, num_clients, mechanism, rounds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Partition data among clients
    data_per_client = len(trainset) // num_clients
    client_dataloaders = [
        DataLoader(Subset(trainset, list(range(i * data_per_client, (i + 1) * data_per_client))), batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]

    test_loader = DataLoader(testset, batch_size=256)

    global_model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    global_acc = []
    client_acc_history = [[] for _ in range(num_clients)]

    for rnd in range(rounds):
        client_models = []

        for i, dataloader in enumerate(client_dataloaders):
            model = SimpleCNN().to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                # Add DP noise
                add_dp_noise(model, scale=1.0 / (epsilon/num_clients), mechanism=mechanism)

                optimizer.step()

            client_models.append(model.state_dict())

            # Evaluate individual client model
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            client_acc_history[i].append(correct / total)

        # Average client models to form new global model
        new_state_dict = {}
        for key in global_model.state_dict().keys():
            new_state_dict[key] = sum([client_model[key] for client_model in client_models]) / num_clients
        global_model.load_state_dict(new_state_dict)

        # Evaluate global model on test set
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        global_acc.append(correct / total)

        # Yield for dynamic plotting
        yield global_acc.copy(), [acc.copy() for acc in client_acc_history]
