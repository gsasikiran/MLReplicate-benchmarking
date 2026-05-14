import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic Data Generation
np.random.seed(42)
X = np.random.rand(1000, 3)  # 1000 samples, 3 features
y = np.clip(X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2], 0, 1)  # EIS target

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class SimpleDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = SimpleDataset(X_train, y_train)
val_dataset = SimpleDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class SimpleNet(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


hidden_units_list = [16, 32, 64]
experiment_data = {
    "learning_rate_variation": {
        "fixed_lr": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "WII": [],
        },
        "scheduler_lr": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "WII": [],
        },
    },
}

for hidden_units in hidden_units_list:
    # Training with fixed learning rate
    model = SimpleNet(hidden_units).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):  # 50 epochs
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

        experiment_data["learning_rate_variation"]["fixed_lr"]["losses"][
            "train"
        ].append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs.view(-1), targets)
                val_loss += loss.item()
                experiment_data["learning_rate_variation"]["fixed_lr"][
                    "predictions"
                ].extend(outputs.cpu().numpy())
                experiment_data["learning_rate_variation"]["fixed_lr"][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        experiment_data["learning_rate_variation"]["fixed_lr"]["losses"]["val"].append(
            val_loss / len(val_loader)
        )
        print(
            f"Fixed LR - Hidden Units: {hidden_units}, Epoch {epoch}: validation_loss = {val_loss:.4f}"
        )

        EIS = np.mean(
            np.array(
                experiment_data["learning_rate_variation"]["fixed_lr"]["predictions"]
            )
            > 0.5
        )
        experiment_data["learning_rate_variation"]["fixed_lr"]["metrics"]["val"].append(
            EIS
        )

        # Worker Impact Index (WII)
        WII = np.mean(
            np.array(
                experiment_data["learning_rate_variation"]["fixed_lr"]["predictions"]
            )
            > 0.5
        )
        experiment_data["learning_rate_variation"]["fixed_lr"]["WII"].append(WII)

    # Training with learning rate scheduler
    model = SimpleNet(hidden_units).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(50):  # 50 epochs
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

        experiment_data["learning_rate_variation"]["scheduler_lr"]["losses"][
            "train"
        ].append(train_loss / len(train_loader))
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs.view(-1), targets)
                val_loss += loss.item()
                experiment_data["learning_rate_variation"]["scheduler_lr"][
                    "predictions"
                ].extend(outputs.cpu().numpy())
                experiment_data["learning_rate_variation"]["scheduler_lr"][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        experiment_data["learning_rate_variation"]["scheduler_lr"]["losses"][
            "val"
        ].append(val_loss / len(val_loader))
        print(
            f"ScheduLer LR - Hidden Units: {hidden_units}, Epoch {epoch}: validation_loss = {val_loss:.4f}"
        )

        EIS = np.mean(
            np.array(
                experiment_data["learning_rate_variation"]["scheduler_lr"][
                    "predictions"
                ]
            )
            > 0.5
        )
        experiment_data["learning_rate_variation"]["scheduler_lr"]["metrics"][
            "val"
        ].append(EIS)

        # Worker Impact Index (WII)
        WII = np.mean(
            np.array(
                experiment_data["learning_rate_variation"]["scheduler_lr"][
                    "predictions"
                ]
            )
            > 0.5
        )
        experiment_data["learning_rate_variation"]["scheduler_lr"]["WII"].append(WII)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
