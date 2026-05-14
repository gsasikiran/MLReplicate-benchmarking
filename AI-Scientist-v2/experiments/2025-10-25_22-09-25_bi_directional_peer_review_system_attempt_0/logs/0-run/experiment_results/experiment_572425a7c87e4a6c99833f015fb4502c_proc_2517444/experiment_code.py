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

# Generate synthetic datasets
np.random.seed(0)
num_samples = 1000

# Dataset 1: Linear
features1 = np.random.rand(
    num_samples, 3
)  # e.g., thoroughness, constructiveness, acceptance rate
labels1 = features1 @ np.array([2.0, 1.5, 3.0]) + np.random.normal(
    0, 0.1, num_samples
)  # Linear with noise

# Dataset 2: Polynomial
features2 = np.random.rand(num_samples, 3)
labels2 = (
    features2[:, 0] ** 2 + features2[:, 1] ** 2 + features2[:, 2] ** 2
) + np.random.normal(
    0, 0.1, num_samples
)  # Quadratic relationship

# Dataset 3: High Noise
features3 = np.random.rand(num_samples, 3)
labels3 = np.random.rand(num_samples) + np.random.normal(
    0, 0.5, num_samples
)  # High noise

datasets = [
    ("linear_data", features1, labels1),
    ("polynomial_data", features2, labels2),
    ("high_noise_data", features3, labels3),
]

# Prepare experiment data storage
experiment_data = {"multi_synthetic_dataset_performance": {}}


# Define simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Hyperparameter tuning for number of hidden units
hidden_units_list = [16, 32, 48, 64]
num_epochs = 10

for dataset_name, features, labels in datasets:
    print(f"\nTraining on dataset: {dataset_name}")

    # Prepare DataLoader
    X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    experiment_data["multi_synthetic_dataset_performance"][dataset_name] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    for hidden_units in hidden_units_list:
        print(f"\nTraining with hidden units: {hidden_units}")
        # Initialize model
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
            experiment_data["multi_synthetic_dataset_performance"][dataset_name][
                "losses"
            ]["train"].append(avg_loss)

            # Calculate RQS (for simplicity using average here)
            rqs = 1 - avg_loss  # Placeholder for actual RQS
            experiment_data["multi_synthetic_dataset_performance"][dataset_name][
                "metrics"
            ]["train"].append(rqs)

            print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}, RQS = {rqs:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
