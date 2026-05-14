import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic Data Generation
np.random.seed(42)
X = np.random.rand(1000, 3)  # 1000 samples, 3 features
y = np.clip(X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2], 0, 1)  # EIS target

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class SimpleDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class SimpleNet(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Scaling Methods
scalers = {"original": None, "minmax": MinMaxScaler(), "standardized": StandardScaler()}

# Initialize experiment data
experiment_data = {
    "feature_scaling": {
        "original": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "minmax": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "standardized": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

for scale_type, scaler in scalers.items():
    if scaler:
        # Scale the data
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        X_train_scaled, X_val_scaled = X_train, X_val

    train_dataset = SimpleDataset(X_train_scaled, y_train)
    val_dataset = SimpleDataset(X_val_scaled, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Hyperparameter tuning for number of hidden units
    hidden_units_list = [16, 32, 64]

    for hidden_units in hidden_units_list:
        model = SimpleNet(hidden_units).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training Loop
        for epoch in range(50):  # 50 epochs
            model.train()
            train_loss = 0.0
            for features, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.view(-1), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            experiment_data["feature_scaling"][scale_type]["losses"]["train"].append(
                train_loss / len(train_loader)
            )

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    outputs = model(features)
                    loss = criterion(outputs.view(-1), targets)
                    val_loss += loss.item()
                    experiment_data["feature_scaling"][scale_type][
                        "predictions"
                    ].extend(outputs.cpu().numpy())
                    experiment_data["feature_scaling"][scale_type][
                        "ground_truth"
                    ].extend(targets.cpu().numpy())

            experiment_data["feature_scaling"][scale_type]["losses"]["val"].append(
                val_loss / len(val_loader)
            )
            print(
                f"Scale Type: {scale_type}, Hidden Units: {hidden_units}, Epoch {epoch}: validation_loss = {val_loss:.4f}"
            )

            # Calculate and store Economic Impact Score (EIS)
            EIS = np.mean(
                np.array(experiment_data["feature_scaling"][scale_type]["predictions"])
                > 0.5
            )
            WII = np.mean(
                np.array(experiment_data["feature_scaling"][scale_type]["ground_truth"])
            )  # Placeholder for WII
            experiment_data["feature_scaling"][scale_type]["metrics"]["val"].append(
                (EIS, WII)
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
