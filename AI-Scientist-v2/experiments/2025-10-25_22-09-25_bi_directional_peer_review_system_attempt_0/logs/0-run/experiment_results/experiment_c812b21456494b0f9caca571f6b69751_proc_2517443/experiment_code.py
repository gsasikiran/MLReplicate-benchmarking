import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Handle GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic dataset
np.random.seed(0)
num_samples = 1000
features = np.random.rand(
    num_samples, 3
)  # e.g., thoroughness, constructiveness, acceptance rate
labels = np.random.rand(num_samples)  # RQS values


# Prepare DataLoader
def create_dataloader(input_features, labels):
    X_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=True)


# Define simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_features, hidden_units):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Prepare data storage for ablation study
experiment_data = {
    "ablation_study": {
        "full_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "thoroughness_ablation": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "constructiveness_ablation": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "acceptance_rate_ablation": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

hidden_units = 32
num_epochs = 10


def train_model(input_features, label_data, ablation_key):
    train_loader = create_dataloader(input_features, label_data)

    model = SimpleNN(input_features.shape[1], hidden_units).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in zip(("features", "labels"), batch)}
            optimizer.zero_grad()
            outputs = model(batch["features"])
            loss = criterion(outputs.squeeze(), batch["labels"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        rqs = 1 - avg_loss  # Placeholder for actual RQS
        experiment_data["ablation_study"][ablation_key]["losses"]["train"].append(
            avg_loss
        )
        experiment_data["ablation_study"][ablation_key]["metrics"]["train"].append(rqs)

        print(
            f"{ablation_key} | Epoch {epoch+1}: loss = {avg_loss:.4f}, RQS = {rqs:.4f}"
        )


# Training with the full dataset
train_model(features, labels, "full_dataset")

# Ablation: Removing each feature one by one
for i in range(3):
    ablation_key = [
        "thoroughness_ablation",
        "constructiveness_ablation",
        "acceptance_rate_ablation",
    ][i]
    ablated_features = np.delete(features, i, axis=1)  # Remove one feature
    train_model(ablated_features, labels, ablation_key)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
