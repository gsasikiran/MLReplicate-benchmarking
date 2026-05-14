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
job_displacement = np.random.uniform(0, 1, data_size)  # Job displacement rates
wage_change = np.random.uniform(-1, 1, data_size)  # Wage fluctuations
retraining_access = np.random.uniform(0, 1, data_size)  # Access to retraining programs
WIS = (
    0.3 * job_displacement + 0.5 * (1 - wage_change) + 0.2 * retraining_access
)  # WIS calculation

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


# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Prepare experiment data storage
experiment_data = {
    "loss_function_ablation": {
        "synthetic_worker_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# List of loss functions and their criteria
loss_functions = {"MSE": nn.MSELoss(), "MAE": nn.L1Loss(), "Huber": nn.SmoothL1Loss()}
learning_rate = 0.001
num_epochs = 50

# Training loop for different loss functions
for loss_name, criterion in loss_functions.items():
    model = SimpleNN().to(device)

    print(f"\nTraining with loss function: {loss_name}")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            features, targets = batch
            features, targets = features.to(device), targets.to(device)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        experiment_data["loss_function_ablation"]["synthetic_worker_data"]["losses"][
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
                experiment_data["loss_function_ablation"]["synthetic_worker_data"][
                    "predictions"
                ].extend(outputs.squeeze().cpu().numpy())
                experiment_data["loss_function_ablation"]["synthetic_worker_data"][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data["loss_function_ablation"]["synthetic_worker_data"]["losses"][
            "val"
        ].append(val_loss)

        # Calculate Worker Impact Score (WIS) and track it
        WIS = 1 - val_loss  # Placeholder for systematic calculation
        experiment_data["loss_function_ablation"]["synthetic_worker_data"]["metrics"][
            "val"
        ].append(WIS)

        print(
            f"Loss Function = {loss_name}, Epoch {epoch + 1}: training_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, WIS = {WIS:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
