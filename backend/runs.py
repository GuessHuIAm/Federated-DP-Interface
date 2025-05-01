"""
This script is used to run the federated learning simulation for user tasks and saves the results to runs subfolder.
"""
from dp_fl_simulation import HeartAttackDataset, HeartAttackMLP, run_dp_federated_learning
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, Subset
import yaml

# Read in yaml file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

EPSILON = config["epsilon"]
CLIP = config["clip"]
NUM_CLIENTS = config["num_clients"]
MECHANISM = config["mechanism"]
ROUNDS = config["rounds"]
EPOCHS_PER_CLIENT = config["epochs_per_client"]
DELTA = float(config["delta"])

# Create runs folder if it doesn't exist
if not os.path.exists("runs"):
    os.makedirs("runs")

# Create unique run subfolder
def create_run_subfolder():
    run_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()
    run_folder = os.path.join("runs", run_id)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

# Save configuration to run folder
def save_config(run_folder):
    with open(os.path.join(run_folder, "config.yaml"), "w") as f:
        yaml.dump(config, f)

# run_folder = create_run_subfolder()
# save_config(run_folder)

# Run the federated learning simulation
run_dp_federated_learning(EPSILON, CLIP, NUM_CLIENTS, MECHANISM, ROUNDS,
                          epochs_per_client=EPOCHS_PER_CLIENT, delta=DELTA)

# Save the model plots
def save_model_plots(run_folder):
    # Assuming you have a function to generate plots
    # This is a placeholder for the actual plotting code
    pass