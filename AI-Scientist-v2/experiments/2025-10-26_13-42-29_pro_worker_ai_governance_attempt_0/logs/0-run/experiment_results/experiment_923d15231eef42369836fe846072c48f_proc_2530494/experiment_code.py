import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Function to create synthetic datasets with controlled correlation
def create_correlated_datasets(num_samples=1000):
    datasets = {}
    correlation_strengths = [
        0.0,
        0.5,
        1.0,
    ]  # Independent, moderate correlation, high correlation
    for corr in correlation_strengths:
        np.random.seed(42)
        job_satisfaction = np.random.rand(num_samples)
        job_security = job_satisfaction * corr + (1 - corr) * np.random.rand(
            num_samples
        )  # Correlated
        retraining_opportunities = job_satisfaction * corr + (
            1 - corr
        ) * np.random.rand(
            num_samples
        )  # Correlated
        wwbi = (job_satisfaction + job_security + retraining_opportunities) / 3

        X = np.column_stack((job_satisfaction, job_security, retraining_opportunities))
        y = wwbi
        datasets[f"correlation_{corr}"] = (X, y)
    return datasets


# Generate datasets
datasets = create_correlated_datasets()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Experiment data storage
experiment_data = {"Input Feature Correlation Ablation": {}}

# Training and validation on each dataset
for dataset_name, (X, y) in datasets.items():
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    experiment_data["Input Feature Correlation Ablation"][dataset_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        experiment_data["Input Feature Correlation Ablation"][dataset_name]["losses"][
            "train"
        ].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_predictions, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        experiment_data["Input Feature Correlation Ablation"][dataset_name]["losses"][
            "val"
        ].append(avg_val_loss)

        # Calculate and store WWBI
        wwbi_metric = np.mean(val_predictions)
        experiment_data["Input Feature Correlation Ablation"][dataset_name]["metrics"][
            "val"
        ].append(wwbi_metric)

        print(
            f"Dataset: {dataset_name}, Epoch {epoch+1}: train_loss = {avg_train_loss:.4f}, validation_loss = {avg_val_loss:.4f}, WWBI = {wwbi_metric:.4f}"
        )

    experiment_data["Input Feature Correlation Ablation"][dataset_name][
        "predictions"
    ] = val_predictions
    experiment_data["Input Feature Correlation Ablation"][dataset_name][
        "ground_truth"
    ] = val_targets

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
