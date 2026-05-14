import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic Data Generation with Noise
def generate_data_with_noise(noise_level):
    np.random.seed(42)
    X = np.random.rand(1000, 3)  # 1000 samples, 3 features
    y = np.clip(X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2], 0, 1)
    noise = np.random.normal(0, noise_level, size=y.shape)  # Adding Gaussian noise
    y_noisy = np.clip(y + noise, 0, 1)  # Ensure target remains in [0, 1]
    return train_test_split(X, y_noisy, test_size=0.2, random_state=42)


# Dataset Class Definition
class SimpleDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Simple Neural Network Definition
class SimpleNet(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create datasets with different noise levels
datasets = {
    "low_noise": generate_data_with_noise(0.05),
    "medium_noise": generate_data_with_noise(0.1),
    "high_noise": generate_data_with_noise(0.2),
}

# Initialize the experiment data structure
experiment_data = {
    "ablation_study": {
        "low_noise": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "medium_noise": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "high_noise": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

hidden_units_list = [16, 32, 64]

for noise_key, (X_train, X_val, y_train, y_val) in datasets.items():
    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for hidden_units in hidden_units_list:
        model = SimpleNet(hidden_units).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training Loop
        for epoch in range(50):
            model.train()
            train_loss = 0.0
            for features, targets in train_loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.view(-1), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            experiment_data["ablation_study"][noise_key]["losses"]["train"].append(
                train_loss / len(train_loader)
            )

            # Validation
            model.eval()
            val_loss = 0.0
            EIS = 0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    outputs = model(features)
                    loss = criterion(outputs.view(-1), targets)
                    val_loss += loss.item()
                    experiment_data["ablation_study"][noise_key]["predictions"].extend(
                        outputs.cpu().numpy()
                    )
                    experiment_data["ablation_study"][noise_key]["ground_truth"].extend(
                        targets.cpu().numpy()
                    )

            experiment_data["ablation_study"][noise_key]["losses"]["val"].append(
                val_loss / len(val_loader)
            )
            EIS = np.mean(
                np.array(experiment_data["ablation_study"][noise_key]["predictions"])
                > 0.5
            )
            experiment_data["ablation_study"][noise_key]["metrics"]["val"].append(EIS)
            print(
                f"Dataset: {noise_key}, Hidden Units: {hidden_units}, Epoch {epoch}: validation_loss = {val_loss:.4f}, WII = {EIS:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
