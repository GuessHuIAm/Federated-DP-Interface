from dp_fl_simulation import run_dp_federated_learning

# Parameters (modify as needed)
epsilon = 3.0
clip = 1.0
num_clients = 5
mechanism = "Gaussian"
rounds = 10
dp_noise = True

# Consume the generator
last_global_acc = None
for global_acc, _ in run_dp_federated_learning(
    epsilon=epsilon,
    clip=clip,
    num_clients=num_clients,
    mechanism=mechanism,
    rounds=rounds,
):
    last_global_acc = global_acc

# Report final accuracy
if last_global_acc:
    print(f"\n✅ Final Global Accuracy after {rounds} rounds: {last_global_acc[-1]:.4f}")
else:
    print("⚠️ Simulation did not produce any output.")
