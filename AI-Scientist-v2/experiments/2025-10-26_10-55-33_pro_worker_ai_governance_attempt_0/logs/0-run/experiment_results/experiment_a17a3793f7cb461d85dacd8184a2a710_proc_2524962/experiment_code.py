import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

np.random.seed(0)

# Synthetic Data Generation for first dataset
X1 = np.random.rand(1000, 3)
y1 = 0.5 * X1[:, 0] + 0.3 * X1[:, 1] + 0.2 * X1[:, 2] + np.random.normal(0, 0.1, 1000)

# Synthetic Data Generation for second dataset
X2 = np.random.rand(1000, 3) * 2  # Different distribution
y2 = 0.4 * X2[:, 0] + 0.4 * X2[:, 1] + 0.1 * X2[:, 2] + np.random.normal(0, 0.2, 1000)

# Synthetic Data Generation for third dataset
X3 = np.random.rand(1000, 3) ** 2  # Different relationship
y3 = 0.6 * X3[:, 0] + 0.2 * X3[:, 1] + 0.2 * X3[:, 2] + np.random.normal(0, 0.15, 1000)

# Normalize Features
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()
X1 = scaler1.fit_transform(X1)
X2 = scaler2.fit_transform(X2)
X3 = scaler3.fit_transform(X3)

# Train-Test Split
datasets = {"dataset_1": (X1, y1), "dataset_2": (X2, y2), "dataset_3": (X3, y3)}
experiment_data = {"ablation_multiple_datasets": {}}

# Convert to PyTorch tensors and define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Hyperparameter tuning for batch size
batch_sizes = [16, 32, 64]
epochs = 100

for name, (X, y) in datasets.items():
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Store metrics and losses for this dataset
    experiment_data["ablation_multiple_datasets"][name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for batch_size in batch_sizes:
        model = SimpleNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create data loaders
        train_data = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).to(device),
            torch.tensor(y_train, dtype=torch.float32).to(device),
        )
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            model.train()
            for data in train_loader:
                inputs, targets = data
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                train_loss = criterion(outputs, targets)
                train_loss.backward()
                optimizer.step()

            experiment_data["ablation_multiple_datasets"][name]["losses"][
                "train"
            ].append(train_loss.item())

            model.eval()
            with torch.no_grad():
                val_outputs = model(
                    torch.tensor(X_val, dtype=torch.float32).to(device)
                ).squeeze()
                val_loss = criterion(
                    val_outputs, torch.tensor(y_val, dtype=torch.float32).to(device)
                )
                experiment_data["ablation_multiple_datasets"][name]["losses"][
                    "val"
                ].append(val_loss.item())

            PWIS = 1 - val_loss.item()
            experiment_data["ablation_multiple_datasets"][name]["metrics"][
                "val"
            ].append(PWIS)

            print(
                f"{name} - Batch Size {batch_size}, Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss.item():.4f}, Validation Loss = {val_loss.item():.4f}, PWIS = {PWIS:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
