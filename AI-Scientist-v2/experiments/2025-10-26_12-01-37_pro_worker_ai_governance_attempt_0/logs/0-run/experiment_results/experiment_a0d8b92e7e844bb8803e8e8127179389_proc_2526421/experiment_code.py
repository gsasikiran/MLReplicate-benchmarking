import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic Data Generation Function
def generate_synthetic_data(samples, features, noise_level=0):
    X = np.random.rand(samples, features)
    y = np.clip(X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2], 0, 1) + np.random.normal(
        0, noise_level, samples
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Datasets
datasets = {
    "dataset_1": generate_synthetic_data(1000, 3, noise_level=0.0),  # No noise
    "dataset_2": generate_synthetic_data(1000, 5, noise_level=0.1),  # Added noise
    "dataset_3": generate_synthetic_data(
        1000, 10, noise_level=0.2
    ),  # More dimensions and noise
}

experiment_data = {"ablation_study": {}}


# Dataset Class Definition
class SimpleDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Model definition
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


hidden_units_list = [16, 32, 64]

for dataset_name, (X_train, X_val, y_train, y_val) in datasets.items():
    experiment_data["ablation_study"][dataset_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for hidden_units in hidden_units_list:
        model = SimpleNet(X_train.shape[1], hidden_units).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training Loop
        for epoch in range(50):  # 50 epochs
            model.train()
            train_loss = 0.0
            for features, targets in DataLoader(
                SimpleDataset(X_train, y_train), batch_size=32, shuffle=True
            ):
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.view(-1), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            experiment_data["ablation_study"][dataset_name]["losses"]["train"].append(
                train_loss / len(X_train) * 32
            )

            # Validation
            model.eval()
            val_loss = 0.0
            predictions, ground_truth = [], []
            with torch.no_grad():
                for features, targets in DataLoader(
                    SimpleDataset(X_val, y_val), batch_size=32, shuffle=False
                ):
                    features, targets = features.to(device), targets.to(device)
                    outputs = model(features)
                    loss = criterion(outputs.view(-1), targets)
                    val_loss += loss.item()
                    predictions.extend(outputs.cpu().numpy())
                    ground_truth.extend(targets.cpu().numpy())
            experiment_data["ablation_study"][dataset_name]["losses"]["val"].append(
                val_loss / len(X_val) * 32
            )
            experiment_data["ablation_study"][dataset_name]["predictions"].extend(
                predictions
            )
            experiment_data["ablation_study"][dataset_name]["ground_truth"].extend(
                ground_truth
            )

            print(
                f"{dataset_name}, Hidden Units: {hidden_units}, Epoch {epoch}: validation_loss = {val_loss:.4f}"
            )

            # Worker Impact Index (WII)
            WII = np.mean(np.array(predictions) > 0.5)
            experiment_data["ablation_study"][dataset_name]["metrics"]["val"].append(
                WII
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
