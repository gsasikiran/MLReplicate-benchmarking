# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 1)
y = 3 * X.squeeze() + np.random.normal(0, 0.5, X.shape[0])

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Function to convert to tensors
def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(
        y, dtype=torch.float32
    ).to(device)


# Prepare data for different scaling approaches
data_settings = {
    "original": (X_train, X_val, y_train, y_val),
    "standardized": (
        StandardScaler().fit_transform(X_train),
        StandardScaler().fit_transform(X_val),
        y_train,
        y_val,
    ),
    "minmax": (
        MinMaxScaler().fit_transform(X_train),
        MinMaxScaler().fit_transform(X_val),
        y_train,
        y_val,
    ),
}


# Define a simple Bayesian linear regression model
class BayesianLinearRegression(nn.Module):
    def __init__(self):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Experiment data
experiment_data = {
    "feature_scale_investigation": {
        "original": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "standardized": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "minmax": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# Hyperparameter tuning for momentum
momentum_values = [0.0, 0.5, 0.9]
for scale_type, (
    X_train_scaled,
    X_val_scaled,
    y_train_scaled,
    y_val_scaled,
) in data_settings.items():
    X_train_tensor, y_train_tensor = to_tensor(X_train_scaled, y_train_scaled)
    X_val_tensor, y_val_tensor = to_tensor(X_val_scaled, y_val_scaled)

    for momentum in momentum_values:
        print(f"Training with momentum: {momentum} and scaling: {scale_type}")

        model = BayesianLinearRegression().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=momentum)

        for epoch in range(100):
            model.train()

            # Forward pass
            predictions = model(X_train_tensor)
            train_loss = criterion(predictions.squeeze(), y_train_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor)
                val_loss = criterion(val_predictions.squeeze(), y_val_tensor)

            # Save metrics
            experiment_data["feature_scale_investigation"][scale_type]["losses"][
                "train"
            ].append(train_loss.item())
            experiment_data["feature_scale_investigation"][scale_type]["losses"][
                "val"
            ].append(val_loss.item())
            experiment_data["feature_scale_investigation"][scale_type][
                "predictions"
            ].append(val_predictions.cpu().numpy())
            experiment_data["feature_scale_investigation"][scale_type][
                "ground_truth"
            ].append(y_val)

            print(
                f"Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
            )

# Reliability measure (simple demonstration)
for scale_type in experiment_data["feature_scale_investigation"]:
    val_predictions = np.concatenate(
        experiment_data["feature_scale_investigation"][scale_type]["predictions"]
    )
    reliable_predictions = (val_predictions.flatten() >= 2.5) & (
        val_predictions.flatten() <= 3.5
    )
    reliability_measure = np.mean(reliable_predictions)
    experiment_data["feature_scale_investigation"][scale_type]["metrics"]["val"].append(
        reliability_measure
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
