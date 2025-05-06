import torch
import torch.nn as nn
import pandas as pd
import logging
import pathlib
import random
import math
from torch.utils.data import Dataset, DataLoader, random_split
import kagglehub
from typing import Union

torch.manual_seed(0) # For reproducibility
random.seed(0)

def _ensure_cardio_csv() -> pathlib.Path:
    # Check if the dataset is already downloaded from Kaggle
    cache_dir = kagglehub.dataset_download("kamilpytlak/personal-key-indicators-of-heart-disease")
    cache_dir = pathlib.Path(cache_dir)
    
    # Search recursively for the .csv file
    matches = list(cache_dir.rglob("heart_2020_cleaned.csv"))
    if not matches:
        raise FileNotFoundError("heart_2020_cleaned.csv not found in Kaggle bundle")
    
    return matches[0]  # Return first match

def auto_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numerical using integer encoding."""
    df = df.copy()
    for col in df.columns:
        try:
            if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
                uniques = df[col].dropna().unique()
                mapping = {val: idx for idx, val in enumerate(uniques)}
                df[col] = df[col].map(mapping)
        except Exception as e:
            logging.warning(f"Failed to encode column {col}: {e}")
            continue
    return df.astype("float32")


class CardioDataset(Dataset):
    """Dataset for heart disease prediction."""
    def __init__(self, csv_path: Union[str, pathlib.Path]):
        # Read in the dataset
        df = pd.read_csv(csv_path, sep=",")
        print(df.shape)

        # Subsample the dataset to 10,000 rows
        df = df.sample(n=10000, random_state=0).reset_index(drop=True)

        # Preprocess the dataset
        df = auto_encode_categoricals(df)
        self.y = df["HeartDisease"].to_numpy(dtype="float32")
        self.X = (
            df.drop(columns=["HeartDisease"])
              .astype("float32")
              .to_numpy()
        )
        self.X = (self.X - self.X.mean(0)) / (self.X.std(0) + 1e-6)

    def __len__(self):  return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class CardioMLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 1),     # logit
        )
    def forward(self, x):
        return self.net(x).squeeze()

# Add Gaussian or Laplace noise to model gradients
def add_dp_noise(model, scale, mechanism="Gaussian"):
    for param in model.parameters():
        if param.grad is None:
            continue
        if mechanism == "Gaussian":
            noise = torch.normal(0, scale, size=param.grad.shape).to(param.device)
        else:
            noise = torch.distributions.Laplace(0, scale).sample(param.grad.shape).to(param.device)
        param.grad += noise

# Federated Learning Simulation
def run_dp_federated_learning(epsilon, clip, num_clients, mechanism, rounds, epochs_per_client=5, delta = 1e-5, dp_noise=False):
    logging.info("Starting DP Federated Learning Simulation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CardioDataset(_ensure_cardio_csv())

    # Train/Test Split
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=64)
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"Number of clients: {num_clients}, Rounds: {rounds}, Epsilon: {epsilon}, Mechanism: {mechanism}")
    print(f"Clip: {clip}, Delta: {delta}, Epochs per client: {epochs_per_client}")

    # Initiate global model
    input_dim = dataset.X.shape[1]
    global_model = CardioMLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Bookkeeping for global and client accuracy
    global_acc = []
    client_acc_history = [[] for _ in range(num_clients)]

    # Split train dataset among clients
    rows_pc = len(train_dataset) // num_clients
    splits = [rows_pc] * (num_clients - 1) + [len(train_dataset) - rows_pc * (num_clients - 1)]
    subsets = random_split(train_dataset, splits)
    client_dataloaders = [DataLoader(s, batch_size=64, shuffle=True) for s in subsets]

    # Initial evaluation
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = global_model(features)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
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
            model = CardioMLP(input_dim).to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            local_loss = 0
            for _ in range(epochs_per_client):
                for features, labels in dataloader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    # Add DP noise
                    if dp_noise:
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
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Round {rnd+1}, Client {i}: Local Accuracy = {correct/total:.4f}, Avg Loss = {local_loss/len(test_loader):.4f}")
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
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        global_acc.append(correct / total)
        print(f"Round {rnd+1}: Global Accuracy: {correct / total:.4f}")

    return global_acc, client_acc_history
        # yield global_acc.copy(), [acc.copy() for acc in client_acc_history]