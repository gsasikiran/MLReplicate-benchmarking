import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Function to generate synthetic datasets
def generate_datasets():
    # Independent features
    np.random.seed(42)
    X1 = np.random.rand(1000, 3)  # Independent
    y1 = np.clip(X1[:, 0] + 0.5 * X1[:, 1] - 0.2 * X1[:, 2], 0, 1)

    # Perfectly correlated features
    X2 = np.random.rand(1000, 1)  # One random feature
    X2 = np.hstack((X2, X2, X2))  # Making them perfectly correlated
    y2 = np.clip(X2[:, 0] + 0.5 * X2[:, 1] - 0.2 * X2[:, 2], 0, 1)

    # Perfectly anti-correlated features
    X3 = np.random.rand(1000, 1)
    X3 = np.hstack((X3, 1 - X3, X3))  # Perfect anti-correlation
    y3 = np.clip(X3[:, 0] + 0.5 * X3[:, 1] - 0.2 * X3[:, 2], 0, 1)

    return [(X1, y1), (X2, y2), (X3, y3)]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

datasets = generate_datasets()
experiment_data = {"input_feature_correlation_ablation": {}}


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

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


hidden_units = 32  # Fixed for simplicity in this ablation study

for idx, (X, y) in enumerate(datasets):
    dataset_name = f"dataset_{idx + 1}"
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = SimpleDataset(X_train, y_train)
    val_dataset = SimpleDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleNet(hidden_units).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    experiment_data["input_feature_correlation_ablation"][dataset_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

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

        experiment_data["input_feature_correlation_ablation"][dataset_name]["losses"][
            "train"
        ].append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs.view(-1), targets)
                val_loss += loss.item()
                experiment_data["input_feature_correlation_ablation"][dataset_name][
                    "predictions"
                ].extend(outputs.cpu().numpy())
                experiment_data["input_feature_correlation_ablation"][dataset_name][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        experiment_data["input_feature_correlation_ablation"][dataset_name]["losses"][
            "val"
        ].append(val_loss / len(val_loader))
        EIS = np.mean(
            np.array(
                experiment_data["input_feature_correlation_ablation"][dataset_name][
                    "predictions"
                ]
            )
            > 0.5
        )
        experiment_data["input_feature_correlation_ablation"][dataset_name]["metrics"][
            "val"
        ].append(EIS)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
