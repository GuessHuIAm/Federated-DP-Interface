import streamlit as st
import matplotlib.pyplot as plt
from dp_fl_simulation import run_dp_federated_learning

st.title("Differentially Private Federated Learning Explorer")

# Sidebar for parameters
epsilon = st.sidebar.slider("Privacy parameter ε", 0.0, 0.1, 10.0, 0.1)
clip = st.sidebar.slider("Clipping norm", 0.0, 5.0, 1.0, 0.1)
num_clients = st.sidebar.slider("Number of clients", 1, 20, 5)
mechanism = st.sidebar.selectbox("DP Mechanism", ["Gaussian", "Laplace"])
rounds = st.sidebar.slider("Number of communication rounds", 1, 100, 5)

# Placeholders for dynamic UI
global_plot_placeholder = st.empty()
client_plot_placeholder = st.empty()
round_status = st.empty()
progress_bar = st.progress(0)

# Run simulation
if st.button("Run Simulation"):
    st.write("Training in progress...")

    for round_num, (global_acc, client_acc) in enumerate(
        run_dp_federated_learning(epsilon, clip, num_clients, mechanism, rounds)
    ):
        # Update round status text
        round_status.info(f"Round {round_num + 1} of {rounds}")

        # Update progress bar
        progress_bar.progress((round_num + 1) / rounds)

        # Global accuracy plot
        fig, ax = plt.subplots()
        ax.plot(global_acc, label="Global Accuracy")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Accuracy")
        ax.set_title("Global Accuracy")
        ax.legend()
        global_plot_placeholder.pyplot(fig)

        # Per-client accuracy plot
        fig, ax = plt.subplots()
        for i, acc in enumerate(client_acc):
            ax.plot(acc, label=f"Client {i+1}")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Client Accuracy")
        ax.legend()
        client_plot_placeholder.pyplot(fig)

    round_status.success("Training complete!")
