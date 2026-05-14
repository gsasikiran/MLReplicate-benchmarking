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

# Define hyperparameter tuning range for batch sizes
batch_sizes = [16, 32, 64, 128]

# Prepare data storage for experiment
experiment_data = {"batch_size_tuning": {}}


# Define simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


num_epochs = 10

# Hyperparameter tuning with different batch sizes
for batch_size in batch_sizes:
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Store results for this batch size
    experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

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
        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["losses"][
            "train"
        ].append(avg_loss)

        # Calculate RQS (for simplicity using average here)
        rqs = 1 - avg_loss
        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["metrics"][
            "train"
        ].append(rqs)

        print(
            f"Batch Size {batch_size}, Epoch {epoch+1}: loss = {avg_loss:.4f}, RQS = {rqs:.4f}"
        )

    # Store predictions and ground truth
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
            "predictions"
        ].extend(predictions)
        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
            "ground_truth"
        ].extend(labels)

# Convert lists to numpy arrays
for batch_size in batch_sizes:
    for metric in ["metrics", "losses"]:
        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][metric] = {
            k: np.array(v)
            for k, v in experiment_data["batch_size_tuning"][
                f"batch_size_{batch_size}"
            ][metric].items()
        }
    experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["predictions"] = (
        np.array(
            experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
                "predictions"
            ]
        )
    )
    experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["ground_truth"] = (
        np.array(
            experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
                "ground_truth"
            ]
        )
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
