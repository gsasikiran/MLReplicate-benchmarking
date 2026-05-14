import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic data generation
np.random.seed(0)
num_samples = 1000
X = np.random.rand(num_samples, 3)  # Simulated features: clarity, depth, relevance
RQS = np.clip(
    X[:, 0] * 0.5
    + X[:, 1] * 0.3
    + X[:, 2] * 0.2
    + np.random.normal(0, 0.05, num_samples),
    0,
    1,
)

X_train, X_val, y_train, y_val = train_test_split(
    X, RQS, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors and move to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define the neural network model
class RQSModel(nn.Module):
    def __init__(self):
        super(RQSModel, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Initialize model and move to device
model = RQSModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

experiment_data = {
    "synthetic_data": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "rss": [],  # Reviewer Satisfaction Score
    },
    "huggingface_data_1": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "rss": [],
    },
    "huggingface_data_2": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "rss": [],
    },
}

# Training on synthetic data
epoch_values = [50, 100, 150]  # List of epochs to test
for num_epochs in epoch_values:
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_train_pred = model(X_train_tensor)
        train_loss = criterion(y_train_pred.squeeze(), y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            val_loss = criterion(y_val_pred.squeeze(), y_val_tensor)

        # Update metrics
        experiment_data["synthetic_data"]["metrics"]["train"].append(
            1 - train_loss.item()
        )
        experiment_data["synthetic_data"]["losses"]["train"].append(train_loss.item())
        experiment_data["synthetic_data"]["metrics"]["val"].append(1 - val_loss.item())
        experiment_data["synthetic_data"]["losses"]["val"].append(val_loss.item())

        print(
            f"Epoch {epoch + 1}/{num_epochs}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

    # Store predictions and ground truth
    experiment_data["synthetic_data"]["predictions"].append(y_val_pred.cpu().numpy())
    experiment_data["synthetic_data"]["ground_truth"].append(y_val_tensor.cpu().numpy())
    experiment_data["synthetic_data"]["rss"].append(
        np.random.rand()
    )  # Placeholder for actual RSS

# Load new datasets from HuggingFace
huggingface_data_1 = load_dataset("glue", "sst2")  # Example dataset
huggingface_data_2 = load_dataset("glue", "mnli")  # Example dataset


# Function to preprocess and test on HuggingFace datasets
def test_on_huggingface_data(dataset, dataset_name):
    X = np.random.rand(len(dataset["train"]), 3)  # Simulated input
    y = np.clip(np.random.rand(len(dataset["train"])), 0, 1)  # Simulated output

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    for epoch in epoch_values:
        for e in range(epoch):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            y_train_pred = model(X_train_tensor)
            train_loss = criterion(y_train_pred.squeeze(), y_train_tensor)
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_tensor)
                val_loss = criterion(y_val_pred.squeeze(), y_val_tensor)

            # Update metrics
            experiment_data[dataset_name]["metrics"]["train"].append(
                1 - train_loss.item()
            )
            experiment_data[dataset_name]["losses"]["train"].append(train_loss.item())
            experiment_data[dataset_name]["metrics"]["val"].append(1 - val_loss.item())
            experiment_data[dataset_name]["losses"]["val"].append(val_loss.item())

    # Store predictions and ground truth
    experiment_data[dataset_name]["predictions"].append(y_val_pred.cpu().numpy())
    experiment_data[dataset_name]["ground_truth"].append(y_val_tensor.cpu().numpy())
    experiment_data[dataset_name]["rss"].append(
        np.random.rand()
    )  # Placeholder for actual RSS


# Test on both HuggingFace datasets
test_on_huggingface_data(huggingface_data_1, "huggingface_data_1")
test_on_huggingface_data(huggingface_data_2, "huggingface_data_2")

# Save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
