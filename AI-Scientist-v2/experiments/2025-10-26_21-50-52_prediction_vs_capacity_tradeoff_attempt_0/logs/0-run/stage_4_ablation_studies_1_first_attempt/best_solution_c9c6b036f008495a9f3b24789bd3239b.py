import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic data generation
np.random.seed(42)
num_samples = 1000
num_features = 10
X = np.random.randn(num_samples, num_features).astype(np.float32)
y = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome

# Create Dataset and DataLoader
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model definition with varying architecture
class SimpleNN(nn.Module):
    def __init__(self, hidden_sizes):
        super(SimpleNN, self).__init__()
        layers = []
        input_size = num_features

        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size

        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter tuning setup
epoch_list = [5, 10, 20, 30]  # Different values for num_epochs

# Experiment data dict for ablation study
experiment_data = {
    "ablation_model_architecture": {
        "synthetic_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Varying configurations for hidden layers
hidden_layer_configs = [
    [16],  # One layer with 16 neurons
    [32],  # One layer with 32 neurons
    [64],  # One layer with 64 neurons
    [16, 16],  # Two layers of 16 neurons
    [32, 32],  # Two layers of 32 neurons
    [64, 32],  # One layer of 64 neurons and one layer of 32 neurons
]

for hidden_sizes in hidden_layer_configs:
    for num_epochs in epoch_list:
        model = SimpleNN(hidden_sizes).to(
            device
        )  # Reinitialize the model for each configuration
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_correct += (predicted == batch_y).sum().item()
                total_samples += batch_y.size(0)

            train_loss = epoch_loss / len(data_loader)
            train_accuracy = total_correct / total_samples
            experiment_data["ablation_model_architecture"]["synthetic_dataset"][
                "losses"
            ]["train"].append(train_loss)
            experiment_data["ablation_model_architecture"]["synthetic_dataset"][
                "metrics"
            ]["train"].append(train_accuracy)

            # Print details
            print(
                f"Config: {hidden_sizes}, Epoch {epoch + 1} of {num_epochs}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
