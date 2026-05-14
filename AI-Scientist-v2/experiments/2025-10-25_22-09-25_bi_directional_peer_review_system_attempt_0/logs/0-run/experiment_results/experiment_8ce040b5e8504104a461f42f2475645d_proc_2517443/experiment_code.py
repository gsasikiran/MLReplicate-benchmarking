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


# Function to generate synthetic dataset
def generate_dataset(seed, num_samples):
    np.random.seed(seed)
    features = np.random.rand(num_samples, 3)  # 3 features
    labels = np.random.rand(num_samples)  # RQS values
    return features, labels


# Generate multiple synthetic datasets
datasets = []
for seed in range(3):
    features, labels = generate_dataset(seed, 1000)
    datasets.append((features, labels))

# Prepare experiment data storage
experiment_data = {
    "dataset_variability_impact": {
        "dataset_1": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_2": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_3": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


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

for idx, (features, labels) in enumerate(datasets):
    print(f"\nTraining on Dataset {idx + 1}")

    # Prepare DataLoader
    X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for hidden_units in hidden_units_list:
        print(f"\nTraining with hidden units: {hidden_units}")
        # Initialize model with the specified number of hidden units
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
            experiment_data["dataset_variability_impact"][f"dataset_{idx + 1}"][
                "losses"
            ]["train"].append(avg_loss)

            # Calculate RQS (for simplicity using average here)
            rqs = 1 - avg_loss  # Placeholder for actual RQS
            experiment_data["dataset_variability_impact"][f"dataset_{idx + 1}"][
                "metrics"
            ]["train"].append(rqs)

            print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}, RQS = {rqs:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
