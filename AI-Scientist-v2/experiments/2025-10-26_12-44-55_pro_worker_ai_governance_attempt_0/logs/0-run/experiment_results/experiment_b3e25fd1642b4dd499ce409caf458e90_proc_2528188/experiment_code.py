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

# Create synthetic dataset
np.random.seed(42)
data_size = 1000
job_displacement = np.random.uniform(0, 1, data_size)
wage_change = np.random.uniform(-1, 1, data_size)
retraining_access = np.random.uniform(0, 1, data_size)
WIS = 0.3 * job_displacement + 0.5 * (1 - wage_change) + 0.2 * retraining_access

X = np.column_stack((job_displacement, wage_change, retraining_access))
y = WIS

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Dataset class
class WorkerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Create DataLoader
train_dataset = WorkerDataset(X_train, y_train)
val_dataset = WorkerDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Define models with different depths
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepNN3(nn.Module):
    def __init__(self):
        super(DeepNN3, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepNN4(nn.Module):
    def __init__(self):
        super(DeepNN4, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Prepare experiment data storage
experiment_data = {
    "ablation_study": {
        "SimpleNN": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "DeepNN3": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "DeepNN4": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

learning_rates = [0.001, 0.01, 0.0001]
num_epochs = 50


# Training function
def train_model(model_class, model_name):
    for lr in learning_rates:
        model = model_class().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        print(f"Training {model_name} with learning rate: {lr}")
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
            experiment_data["ablation_study"][model_name]["losses"]["train"].append(
                train_loss
            )

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    features, targets = batch
                    features, targets = features.to(device), targets.to(device)

                    outputs = model(features)
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()
                    experiment_data["ablation_study"][model_name]["predictions"].extend(
                        outputs.squeeze().cpu().numpy()
                    )
                    experiment_data["ablation_study"][model_name][
                        "ground_truth"
                    ].extend(targets.cpu().numpy())

            val_loss /= len(val_loader)
            experiment_data["ablation_study"][model_name]["losses"]["val"].append(
                val_loss
            )

            WIS = 1 - val_loss
            experiment_data["ablation_study"][model_name]["metrics"]["val"].append(WIS)

            print(
                f"{model_name}: Learning Rate = {lr}, Epoch {epoch + 1}: training_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, WIS = {WIS:.4f}"
            )


# Train models
train_model(SimpleNN, "SimpleNN")
train_model(DeepNN3, "DeepNN3")
train_model(DeepNN4, "DeepNN4")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
