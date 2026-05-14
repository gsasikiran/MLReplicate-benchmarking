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
features = np.random.rand(num_samples, 3)
labels = np.random.rand(num_samples)

# Prepare DataLoader
X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor, y_tensor)


# Define simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Prepare data storage for ablation study
experiment_data = {
    "impact_of_batch_size": {
        "synthetic_data": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Set parameters for the ablation study
batch_sizes = [16, 32, 64, 128]
hidden_units = 32  # Fixed number of hidden units for the study
num_epochs = 10

for batch_size in batch_sizes:
    print(f"\nTraining with batch size: {batch_size}")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN(hidden_units).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
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
        experiment_data["impact_of_batch_size"]["synthetic_data"]["losses"][
            "train"
        ].append(avg_loss)

        # Calculate RQS (for simplicity using average here)
        rqs = 1 - avg_loss
        experiment_data["impact_of_batch_size"]["synthetic_data"]["metrics"][
            "train"
        ].append(rqs)

        print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}, RQS = {rqs:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
