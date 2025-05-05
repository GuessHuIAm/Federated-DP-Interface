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

# # Run the federated learning simulation
# run_dp_federated_learning(EPSILON, CLIP, NUM_CLIENTS, MECHANISM, ROUNDS,
#                           epochs_per_client=EPOCHS_PER_CLIENT, delta=DELTA,
#                           dp_noise=DP_NOISE)

# Save the model plots
def run():
    # Create subfolder for run of information
    run_folder = create_run_subfolder()
    save_config(run_folder)

    # Run one simulation
    global_acc, client_acc = run_dp_federated_learning(
        EPSILON, CLIP, NUM_CLIENTS, MECHANISM, ROUNDS,
        epochs_per_client=EPOCHS_PER_CLIENT, delta=DELTA,
        dp_noise=DP_NOISE
    )
    
    # Save accuracy histories as json
    with open(os.path.join(run_folder, "global_acc.json"), "w") as f:
        json.dump(global_acc, f)
    with open(os.path.join(run_folder, "client_acc.json"), "w") as f:
        json.dump(client_acc, f)

    # Save the model plots
    plt.figure()
    plt.plot(global_acc, label="Global Model Accuracy")
    plt.title("Global Model Accuracy")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(run_folder, "global_acc.png"))
    plt.close()

    # Plot all the client accuracies on one plot
    plt.figure()
    for i, client_acc_i in enumerate(client_acc):
        plt.plot(client_acc_i, label=f"Client {i+1}", color=plt.cm.tab10(i))
    plt.title("Client Model Accuracies")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(run_folder, "client_acc.png"))
    plt.close()
    print(f"Run completed. Results saved in {run_folder}")

run()