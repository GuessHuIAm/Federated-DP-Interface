import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import random

# Load Heart Attack dataset
class HeartAttackDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        # Drop non-numeric / problematic columns
        data = data.drop(columns=[
            'Patient ID',      # String ID
            'Blood Pressure',  # '120/80' format
            'Country',         # Text
            'Continent',       # Text
            'Hemisphere',      # Text
            'Diet',            # Text
            'Sex'              # Text
        ])

        # Separate features and target
        self.X = data.drop(columns=['Heart Attack Risk']).values.astype(np.float32)
        self.y = data['Heart Attack Risk'].values.astype(np.float32)

        # Normalize features (critical for MLPs)
        self.X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-6)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# Deeper MLP model for tabular data
class HeartAttackMLP(nn.Module):
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
def run_dp_federated_learning(epsilon, clip, num_clients, mechanism, rounds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = HeartAttackDataset("./datasets/heart_attack_prediction_dataset.csv")

    # Split into clients
    data_per_client = len(dataset) // num_clients
    client_dataloaders = [
        DataLoader(Subset(dataset, list(range(i * data_per_client, (i + 1) * data_per_client))),
                   batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]

    # Train/Test Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=64)

    input_dim = dataset.X.shape[1]
    global_model = HeartAttackMLP(input_dim).to(device)
    criterion = nn.BCELoss()

    global_acc = []
    client_acc_history = [[] for _ in range(num_clients)]

    # Initial evaluation before training starts (round 0)
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
    yield global_acc.copy(), [[] for _ in range(num_clients)]

    for rnd in range(rounds):
        client_models = []

        for dataloader in client_dataloaders:
            model = HeartAttackMLP(input_dim).to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Local training: 2 epochs instead of 1
            for epoch in range(2):
                for features, labels in dataloader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    add_dp_noise(model, scale=1.0 / (epsilon / num_clients), mechanism=mechanism)
                    optimizer.step()

            client_models.append(model.state_dict())

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
            print(f"Round {rnd+1}: Global Accuracy = {correct / total:.4f}")

            client_acc_history[client_dataloaders.index(dataloader)].append(correct / total)

        # Aggregate global model
        new_state_dict = {}
        for key in global_model.state_dict().keys():
            new_state_dict[key] = sum([client_model[key] for client_model in client_models]) / num_clients
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

        # Yield after each round
        yield global_acc.copy(), [acc.copy() for acc in client_acc_history]
