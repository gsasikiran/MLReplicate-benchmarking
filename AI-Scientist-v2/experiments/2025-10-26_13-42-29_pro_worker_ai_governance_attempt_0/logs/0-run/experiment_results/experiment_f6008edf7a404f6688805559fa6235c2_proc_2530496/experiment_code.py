import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic dataset creation
np.random.seed(42)
num_samples = 1000
job_satisfaction = np.random.rand(num_samples)
job_security = np.random.rand(num_samples)
retraining_opportunities = np.random.rand(num_samples)
wwbi = (job_satisfaction + job_security + retraining_opportunities) / 3

X = np.column_stack((job_satisfaction, job_security, retraining_opportunities))
y = wwbi


# Function to prepare the data
def prepare_data(X, y):
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
    return train_loader, val_loader, y_val


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


# Ablation study container
experiment_data = {
    "input_feature_scaling": {
        "unscaled": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "standard_scaled": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "minmax_scaled": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# Train and evaluate the model for different scaling methods
scalers = {
    "unscaled": None,
    "standard_scaled": StandardScaler(),
    "minmax_scaled": MinMaxScaler(),
}

for scaling_type, scaler in scalers.items():
    if scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Prepare data loaders
    train_loader, val_loader, val_targets = prepare_data(X_scaled, y)

    # Initialize model, loss, optimizer
    model = SimpleNN().to(device)  # Move model to device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(
                device
            )  # Move input tensors to device

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        experiment_data["input_feature_scaling"][scaling_type]["losses"][
            "train"
        ].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(
                    device
                )  # Move input tensors to device
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        experiment_data["input_feature_scaling"][scaling_type]["losses"]["val"].append(
            avg_val_loss
        )

        # Calculate and store WWBI
        wwbi_metric = np.mean(val_predictions)
        experiment_data["input_feature_scaling"][scaling_type]["metrics"]["val"].append(
            wwbi_metric
        )

        print(
            f"Scaling Type: {scaling_type}, Epoch {epoch+1}: train_loss = {avg_train_loss:.4f}, validation_loss = {avg_val_loss:.4f}, WWBI = {wwbi_metric:.4f}"
        )

    experiment_data["input_feature_scaling"][scaling_type][
        "predictions"
    ] = val_predictions
    experiment_data["input_feature_scaling"][scaling_type]["ground_truth"] = val_targets

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
