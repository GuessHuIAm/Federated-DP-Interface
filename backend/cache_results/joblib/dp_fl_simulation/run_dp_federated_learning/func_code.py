# first line: 91
def run_dp_federated_learning(epsilon, clip, num_clients, mechanism, rounds, epochs_per_client=5, delta = 1e-5, dp_noise=True):
    torch.manual_seed(0) # For reproducibility
    random.seed(0)
    
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
    average_noise_magnitudes = []
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
        round_noise = 0

        # Each client trains its local model
        for i, dataloader in enumerate(client_dataloaders):
            model = CardioMLP(input_dim).to(device)
            model.load_state_dict(global_model.state_dict())
            model.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            local_loss = 0
            client_noise = 0
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
                        noise_norm = add_dp_noise(model, scale=scale, mechanism=mechanism)
                        client_noise += noise_norm
                    optimizer.step()
                    local_loss += loss.item()
            
            scheduler.step()

            client_models.append(model.state_dict())
            client_sizes.append(len(dataloader.dataset))
            
            round_noise += client_noise / (epochs_per_client * len(dataloader))

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

            client_acc_history[i].append(correct / total)
        
        average_noise_magnitudes.append(round_noise / num_clients)

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

    if dp_noise:
        average_noise = sum(average_noise_magnitudes) / len(average_noise_magnitudes)
        return global_acc, client_acc_history, average_noise
    return global_acc, client_acc_history