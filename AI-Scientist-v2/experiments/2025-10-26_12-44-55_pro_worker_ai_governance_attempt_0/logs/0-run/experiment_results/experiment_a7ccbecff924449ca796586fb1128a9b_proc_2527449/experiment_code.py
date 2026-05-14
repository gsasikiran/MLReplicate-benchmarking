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

# Prepare experiment data storage
experiment_data = {
    "hyperparam_tuning": {
        "hidden_layer_size": {},
    }
}

# Hyperparameter tuning for the hidden layer size
hidden_layer_sizes = [5, 10, 20, 30]  # Different sizes to explore
num_epochs = 50

for size in hidden_layer_sizes:
    # Define the model with variable hidden layer size
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(3, size)
            self.fc2 = nn.Linear(size, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize model, loss function, and optimizer
    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Prepare data storage for this hidden layer size
    experiment_data["hyperparam_tuning"]["hidden_layer_size"][size] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Training loop
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
        experiment_data["hyperparam_tuning"]["hidden_layer_size"][size]["losses"][
            "train"
        ].append(train_loss)

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
                experiment_data["hyperparam_tuning"]["hidden_layer_size"][size][
                    "predictions"
                ].extend(outputs.squeeze().cpu().numpy())
                experiment_data["hyperparam_tuning"]["hidden_layer_size"][size][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data["hyperparam_tuning"]["hidden_layer_size"][size]["losses"][
            "val"
        ].append(val_loss)

        # Calculate Worker Impact Score (WIS) and track it
        WIS = 1 - val_loss
        experiment_data["hyperparam_tuning"]["hidden_layer_size"][size]["metrics"][
            "val"
        ].append(WIS)

        print(
            f"Hidden Layer Size {size} - Epoch {epoch + 1}: training_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, WIS = {WIS:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
