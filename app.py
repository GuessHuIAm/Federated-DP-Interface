import streamlit as st
import matplotlib.pyplot as plt
import time
from dp_fl_simulation import run_dp_federated_learning

if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False

st.title("Differentially Private Federated Learning Explorer")

def start_training():
    st.session_state.simulation_running = True

# Sidebar parameters
epsilon = st.sidebar.slider("Privacy parameter ε", 0.0, 10.0, 1.0, 0.1, disabled=st.session_state.simulation_running)
clip = st.sidebar.slider("Clipping norm", 0.0, 5.0, 1.0, 0.1, disabled=st.session_state.simulation_running)
num_clients = st.sidebar.slider("Number of clients", 1, 20, 5, disabled=st.session_state.simulation_running)
mechanism = st.sidebar.selectbox("DP Mechanism", ["Gaussian", "Laplace"], disabled=st.session_state.simulation_running)
rounds = st.sidebar.slider("Number of communication rounds", 1, 100, 5, disabled=st.session_state.simulation_running)

# UI placeholders
global_plot_placeholder = st.empty()
client_plot_placeholder = st.empty()
round_status = st.empty()
client_status = st.empty()
client_progress_bar = st.progress(0)

# Run button
st.button("Run Simulation", on_click=start_training, disabled=st.session_state.simulation_running)

# Training logic
if st.session_state.simulation_running:
    st.write("Training in progress...")

    for round_num, (global_acc, client_acc) in enumerate(
        run_dp_federated_learning(
            epsilon, clip, num_clients, mechanism, rounds,
            client_status=client_status,
            client_progress_bar=client_progress_bar
        )
    ):
        round_status.info(f"Completed Round {round_num} of {rounds}")

        # Global accuracy plot
        fig, ax = plt.subplots()
        ax.plot(range(len(global_acc)), global_acc, label="Global Accuracy")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Accuracy")
        ax.set_title("Global Accuracy")
        ax.legend()
        global_plot_placeholder.pyplot(fig)

        # Per-client accuracy plot
        fig, ax = plt.subplots()
        for i, acc in enumerate(client_acc):
            ax.plot(range(len(acc)), acc, label=f"Client {i+1}")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Client Accuracy")
        ax.legend()
        client_plot_placeholder.pyplot(fig)

    # Cleanup
    round_status.success("Training complete!")
    client_progress_bar.empty()
    st.session_state.simulation_running = False
