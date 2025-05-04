# Update the full federated learning script to use a fixed noise multiplier with privacy accounting

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Subset

# Custom RDP accountant
class SimpleRDPAccountant:
    def __init__(self, alphas=None):
        if alphas is None:
            self.alphas = np.arange(1.1, 10.0, 0.1).tolist() + list(range(10, 100))
        else:
            self.alphas = alphas
        self.steps = 0
        self.total_rdp = np.zeros(len(self.alphas))

    def _compute_rdp(self, q, noise_multiplier, steps):
        if noise_multiplier == 0:
            return np.inf * np.ones_like(self.alphas)
        orders = np.array(self.alphas)
        rdp = np.array([self._compute_rdp_scalar(q, noise_multiplier, order) for order in orders])
        return rdp * steps

    def _compute_rdp_scalar(self, q, sigma, alpha):
        if q == 0:
            return 0
        if sigma == 0:
            return np.inf
        if q == 1.0:
            return alpha / (2 * sigma ** 2)
        return (1 / (alpha - 1)) * np.log(q * ((alpha - 1) / (sigma ** 2)) ** 0.5 + (1 - q))

    def step(self, noise_multiplier, sample_rate):
        self.steps += 1
        self.total_rdp += self._compute_rdp(sample_rate, noise_multiplier, 1)

    def get_epsilon(self, delta):
        epsilons = self.total_rdp - np.log(delta) / (np.array(self.alphas) - 1)
        # print ("Epsilons:", epsilons)
        idx = np.argmin(epsilons)
        return epsilons[idx], self.alphas[idx]

# Dataset class
class HeartAttackDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        data = pd.get_dummies(data, columns=categorical_cols)
        self.X = data.drop(columns=['HeartDisease']).values.astype(np.float32)
        self.y = data['HeartDisease'].values.astype(np.float32)
        self.X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-6)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# MLP model
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

# Add DP noise with fixed noise multiplier
def add_dp_noise(model, scale):
    for param in model.parameters():
        if param.requires_grad:
            noise = torch.normal(0, scale, size=param.grad.shape).to(param.device)
            param.grad += noise

# Federated learning with privacy accountant
def run_dp_federated_learning(epsilon, clip, num_clients, mechanism, rounds, delta=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Heuristic conversion from epsilon to noise multiplier
    noise_multiplier = 1.0 / epsilon  # You can tune this

    # Load dataset
    dataset = HeartAttackDataset("./datasets/heart.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=64)

    input_dim = dataset.X.shape[1]
    global_model = HeartAttackMLP(input_dim).to(device)
    criterion = nn.BCELoss()

    data_per_client = len(train_dataset) // num_clients
    client_dataloaders = [
        DataLoader(Subset(train_dataset, list(range(i * data_per_client, (i + 1) * data_per_client))),
                   batch_size=64, shuffle=True)
        for i in range(num_clients)
    ]

    global_acc = []
    client_acc_history = [[] for _ in range(num_clients)]

    accountant = SimpleRDPAccountant()
    sample_rate = 1.0  # assuming all clients participate each round

    # Initial global evaluation
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
        client_sizes = []

        for i, dataloader in enumerate(client_dataloaders):
            model = HeartAttackMLP(input_dim).to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            for epoch in range(5):
                for features, labels in dataloader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    add_dp_noise(model, noise_multiplier * clip)
                    optimizer.step()
                scheduler.step()

            client_models.append(model.state_dict())
            client_sizes.append(len(dataloader.dataset))

            # Evaluate client model
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for features, labels in dataloader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            client_acc_history[i].append(correct / total)

        # Aggregate client updates
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

        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        # print(f"Round {rnd+1}: Global Accuracy = {correct / total:.4f}")

        yield global_acc.copy(), [acc.copy() for acc in client_acc_history]

    # Final privacy cost
    epsilon_final, best_alpha = accountant.get_epsilon(delta)
    print(f"\nFinal ε after {rounds} rounds: ε = {epsilon_final:.4f} at α = {best_alpha}")
