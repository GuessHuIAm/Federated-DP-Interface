# run_cli.py

from dp_fl_simulation import run_dp_federated_learning

if __name__ == "__main__":
    print("=== DP Federated Learning Simulator ===")
    epsilon = float(input("Enter ε (epsilon): "))
    num_clients = int(input("Enter number of clients: "))
    rounds = int(input("Enter number of rounds: "))

    final_global_acc, _, _ = run_dp_federated_learning(
        epsilon=epsilon,
        clip=1.0,
        num_clients=num_clients,
        mechanism="Gaussian",
        rounds=rounds,
        dp_noise=True
    )

    print(f"\n✅ Final Global Accuracy after {rounds} rounds: {final_global_acc[-1]:.4f}")
