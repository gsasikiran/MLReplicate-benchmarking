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


# Function to train the model
def train_model(X_train, y_train, X_val, y_val, lr):
    train_dataset = WorkerDataset(X_train, y_train)
    val_dataset = WorkerDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SimpleNN(input_size=X_train.shape[1]).to(device)  # Adjust model input size
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    metrics = {"train": [], "val": []}
    losses = {"train": [], "val": []}
    predictions = []
    ground_truth = []

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
        losses["train"].append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features, targets = batch
                features, targets = features.to(device), targets.to(device)

                outputs = model(features)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                predictions.extend(outputs.squeeze().cpu().numpy())
                ground_truth.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        losses["val"].append(val_loss)
        PWIS = 1 - val_loss  # Update Pro-Worker Impact Score
        metrics["val"].append(PWIS)

        print(
            f"Learning Rate = {lr}, Epoch {epoch + 1}: training_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, PWIS = {PWIS:.4f}"
        )

    return metrics, losses, predictions, ground_truth


# Dataset class
class WorkerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Prepare experiment data storage
experiment_data = {
    "ablation_job_displacement": {
        "synthetic_worker_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
    "ablation_wage_change": {
        "synthetic_worker_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
    "ablation_retraining_access": {
        "synthetic_worker_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
}

# List of learning rates to test
learning_rates = [0.001, 0.01, 0.0001]
num_epochs = 50

# Run ablation studies
for feature in range(3):  # 0: job_displacement, 1: wage_change, 2: retraining_access
    print(f"\nAblation Study for Feature Index: {feature}")
    # Prepare sets of features for training
    if feature == 0:  # Omitting job_displacement
        X_temp = X[:, [1, 2]]
    elif feature == 1:  # Omitting wage_change
        X_temp = X[:, [0, 2]]
    else:  # Omitting retraining_access
        X_temp = X[:, [0, 1]]

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y, test_size=0.2, random_state=42
    )

    for lr in learning_rates:
        metrics, losses, predictions, ground_truth = train_model(
            X_train, y_train, X_val, y_val, lr
        )

        if feature == 0:
            experiment_data["ablation_job_displacement"]["synthetic_worker_data"][
                "metrics"
            ]["val"].extend(metrics["val"])
            experiment_data["ablation_job_displacement"]["synthetic_worker_data"][
                "losses"
            ]["val"].extend(losses["val"])
            experiment_data["ablation_job_displacement"]["synthetic_worker_data"][
                "predictions"
            ].extend(predictions)
            experiment_data["ablation_job_displacement"]["synthetic_worker_data"][
                "ground_truth"
            ].extend(ground_truth)
        elif feature == 1:
            experiment_data["ablation_wage_change"]["synthetic_worker_data"]["metrics"][
                "val"
            ].extend(metrics["val"])
            experiment_data["ablation_wage_change"]["synthetic_worker_data"]["losses"][
                "val"
            ].extend(losses["val"])
            experiment_data["ablation_wage_change"]["synthetic_worker_data"][
                "predictions"
            ].extend(predictions)
            experiment_data["ablation_wage_change"]["synthetic_worker_data"][
                "ground_truth"
            ].extend(ground_truth)
        else:
            experiment_data["ablation_retraining_access"]["synthetic_worker_data"][
                "metrics"
            ]["val"].extend(metrics["val"])
            experiment_data["ablation_retraining_access"]["synthetic_worker_data"][
                "losses"
            ]["val"].extend(losses["val"])
            experiment_data["ablation_retraining_access"]["synthetic_worker_data"][
                "predictions"
            ].extend(predictions)
            experiment_data["ablation_retraining_access"]["synthetic_worker_data"][
                "ground_truth"
            ].extend(ground_truth)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
