import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
import math
from datetime import datetime
import hashlib
import os
from torch.utils.data import Dataset, DataLoader, Subset

delta = 1e-5

# Load new Heart Attack dataset
class HeartAttackDataset(Dataset):
    """
    Wrapper for processed dataset
    @csv_file: string path to the CSV file
    """
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        # No need to drop columns in the new dataset
        self.y = data['target'].values.astype(np.float32)
        self.X = data.drop(columns=['target']).values.astype(np.float32)

        # Normalize features
        self.X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-6)
        assert(self.X.shape[0] == data.shape[0])
        assert(self.y.shape[0] == data.shape[0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# Deeper MLP model for tabular data
class HeartAttackMLP(nn.Module):
    """
    A simple MLP model for binary classification
    @input_dim: number of features in the input data
    """
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x)).squeeze()

# Add Gaussian or Laplace noise to model gradients
def add_dp_noise(model, scale, mechanism="Gaussian"):
    for param in model.parameters():
        if param.requires_grad:
            if mechanism == "Gaussian":
                noise = torch.normal(0, scale, size=param.grad.shape).to(param.device)
            else:
                noise = torch.distributions.Laplace(0, scale).sample(param.grad.shape).to(param.device)
            param.grad += noise

# Federated Learning Simulation
def run_dp_federated_learning(epsilon, clip, num_clients, mechanism, rounds, epochs_per_client=5):
    logging.info("Starting DP Federated Learning Simulation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = HeartAttackDataset("./datasets/Heart Attack Data Set.csv")

    # Train/Test Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initiate global model
    input_dim = dataset.X.shape[1]
    global_model = HeartAttackMLP(input_dim).to(device)
    criterion = nn.BCELoss()

    # Bookkeeping for global and client accuracy
    global_acc = []
    client_acc_history = [[] for _ in range(num_clients)]

    # Split train dataset among clients
    data_per_client = len(train_dataset) // num_clients
    client_dataloaders = [
        DataLoader(Subset(train_dataset, list(range(i * data_per_client, (i + 1) * data_per_client))),
                   batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]

    # Initial evaluation
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = global_model(features)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    global_acc.append(correct / total)
    print(f"Initial Global Accuracy: {correct / total:.4f}")
    # yield global_acc.copy(), [[] for _ in range(num_clients)]

    # For each communication round
    for rnd in range(rounds):
        client_models = []
        client_sizes = []

        # Each client trains its local model
        for i, dataloader in enumerate(client_dataloaders):
            model = HeartAttackMLP(input_dim).to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            local_loss = 0
            for _ in range(epochs_per_client):
                for features, labels in dataloader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    T = rounds * epochs_per_client * len(dataloader) 
                    scale = (1 / epsilon) * math.sqrt(2 * math.log(1.25 / delta)) * math.sqrt(T)
                    add_dp_noise(model, scale=scale, mechanism=mechanism)
                    optimizer.step()

                    local_loss += loss.item()
                scheduler.step()

            client_models.append(model.state_dict())
            client_sizes.append(len(dataloader.dataset))

            # Evaluate individual client model
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for features, labels in dataloader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Round {rnd+1}, Client {i}: Local Accuracy = {correct/total:.4f}, Avg Loss = {local_loss/len(dataloader):.4f}")
            client_acc_history[i].append(correct / total)

        # Weighted aggregation
        total_samples = sum(client_sizes)
        new_state_dict = {}
        for key in global_model.state_dict().keys():
            new_state_dict[key] = sum(
                (size / total_samples) * client_model[key]
                for size, client_model in zip(client_sizes, client_models)
            )
        global_model.load_state_dict(new_state_dict)

        # Evaluate global model
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = global_model(features)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        global_acc.append(correct / total)
        print(f"Round {rnd+1}: Global Accuracy: {correct / total:.4f}")

        # yield global_acc.copy(), [acc.copy() for acc in client_acc_history]
