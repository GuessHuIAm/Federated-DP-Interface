"""
This script is used to run the federated learning simulation for user tasks and saves the results to runs subfolder.
"""
from dp_fl_simulation import run_dp_federated_learning
import os
import json
import matplotlib.pyplot as plt
import hashlib
from datetime import datetime
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
DP_NOISE = bool(config["dp_noise"])

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

# Save global and client accuracies to JSON files
def save_accuracies(run_folder, global_acc, client_acc):
    with open(os.path.join(run_folder, "global_acc.json"), "w") as f:
        json.dump(global_acc, f)
    with open(os.path.join(run_folder, "client_acc.json"), "w") as f:
        json.dump(client_acc, f)

# run_folder = create_run_subfolder()
# save_config(run_folder)

# # Run the federated learning simulation
# global_acc, client_acc = run_dp_federated_learning(EPSILON, CLIP, NUM_CLIENTS, MECHANISM, ROUNDS,
#                           epochs_per_client=EPOCHS_PER_CLIENT, delta=DELTA,
#                           dp_noise=DP_NOISE)

# save_accuracies(run_folder, global_acc, client_acc)

var = "clients"
path = f"runs/{var}"
vars = []
global_accs = []

for folder in os.listdir(path):
    # Continue if not a folder
    if not os.path.isdir(os.path.join(path, folder)):
        continue

    # Get config
    config_path = os.path.join(path, folder, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    ROUNDS = config["rounds"]
    EPSILON = config["epsilon"]
    CLIENTS = config["num_clients"]
    if var == "clients":
        var_value = CLIENTS
    elif var == "epsilon":
        var_value = EPSILON
    elif var == "rounds":
        var_value = ROUNDS
    else:
        pass
    vars.append(var_value)

    # Get global accuracy
    json_path = os.path.join(path, folder, "global_acc.json")
    with open(json_path, "r") as f:
        global_acc = list(json.load(f))[-1]
    global_accs.append(global_acc)

# Sort rounds
vars, global_accs = zip(*sorted(zip(vars, global_accs), key=lambda x: x[0]))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(vars, global_accs, label="Global Accuracy")
# Draw curved line
plt.xlabel(var)
plt.ylabel("Accuracy")
if var == "rounds":
    plt.title(f"Rounds (epsilon={EPSILON}, clients={CLIENTS})")
elif var == "epsilon":
    plt.title(f"Epsilon (rounds={ROUNDS}, clients={CLIENTS})")
elif var == "clients":
    plt.title(f"Clients (epsilon={EPSILON}, rounds={ROUNDS})")
else:
    pass
plt.legend()
plt.grid()
plt.savefig(os.path.join(path, "global_accuracy_plot.png"))