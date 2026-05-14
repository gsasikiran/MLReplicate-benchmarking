import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Custom Dataset class
class WorkerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Simple Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.fc(x)


# Function to create synthetic dataset
def create_dataset(data_size, seed):
    np.random.seed(seed)
    job_displacement = np.random.uniform(0, 1, data_size)
    wage_change = np.random.uniform(-1, 1, data_size)
    retraining_access = np.random.uniform(0, 1, data_size)
    WIS = 0.3 * job_displacement + 0.5 * (1 - wage_change) + 0.2 * retraining_access
    return np.column_stack((job_displacement, wage_change, retraining_access)), WIS


# Create multiple synthetic datasets
datasets = {}
datasets["dataset_1"] = create_dataset(1000, seed=42)
datasets["dataset_2"] = create_dataset(1000, seed=24)
datasets["dataset_3"] = create_dataset(1000, seed=99)

# Prepare experiment data storage
experiment_data = {
    "ablation_study": {
        "dataset_1": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_2": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_3": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# List of learning rates to test
learning_rates = [0.001, 0.01, 0.0001]
num_epochs = 50

# Training loop for different datasets
for dataset_name, (features, targets) in datasets.items():
    X_train, X_val, y_train, y_val = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    train_dataset = WorkerDataset(X_train, y_train)
    val_dataset = WorkerDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    for lr in learning_rates:
        model = SimpleNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        print(f"Training {dataset_name} with learning rate: {lr}")
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                features, targets = batch
                features, targets = features.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            experiment_data["ablation_study"][dataset_name]["losses"]["train"].append(
                train_loss
            )

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    features, targets = batch
                    features, targets = features.to(device), targets.to(device)

                    outputs = model(features)
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()
                    experiment_data["ablation_study"][dataset_name][
                        "predictions"
                    ].extend(outputs.squeeze().cpu().numpy())
                    experiment_data["ablation_study"][dataset_name][
                        "ground_truth"
                    ].extend(targets.cpu().numpy())

            val_loss /= len(val_loader)
            experiment_data["ablation_study"][dataset_name]["losses"]["val"].append(
                val_loss
            )

            # Calculate Pro-Worker Impact Score (PWIS) and track it
            PWIS = 1 - val_loss
            experiment_data["ablation_study"][dataset_name]["metrics"]["val"].append(
                PWIS
            )

            print(
                f"{dataset_name}: Learning Rate = {lr}, Epoch {epoch + 1}: training_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, PWIS = {PWIS:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
