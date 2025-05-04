import torch
import torch.nn as nn
import pandas as pd
import logging
import pathlib
import math
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import kagglehub
from typing import Union
import random
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# ... (existing CardioDataset, CardioMLP, and training functions remain unchanged) ...

# Place at the bottom of the file
if __name__ == "__main__":
    epsilons = [0.5, 1.0, 2.0, 3.0, 5.0]
    results = {}

    for eps in epsilons:
        print(f"\nRunning for epsilon = {eps}")
        acc_history = []
        fl_gen = run_dp_federated_learning(
            epsilon=eps,
            clip=0.7,
            num_clients=10,
            mechanism="Gaussian",
            rounds=30,
            epochs_per_client=5,
            delta=1e-5,
            client_frac=0.6
        )

        for global_acc, _ in fl_gen:
            acc_history.append(global_acc[-1])

        results[eps] = acc_history

    # Plotting
    plt.figure(figsize=(10, 6))
    for eps, acc in results.items():
        plt.plot(acc, label=f"ε = {eps}")

    plt.title("Global Accuracy vs Communication Rounds for Different ε")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
