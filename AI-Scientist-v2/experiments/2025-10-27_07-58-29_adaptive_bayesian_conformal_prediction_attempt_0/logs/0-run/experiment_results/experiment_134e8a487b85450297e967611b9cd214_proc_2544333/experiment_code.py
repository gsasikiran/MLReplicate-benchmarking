import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Function to generate synthetic data
def generate_synthetic_data(dimensions, samples=1000):
    X = np.random.rand(samples, dimensions)
    y = 3 * X.sum(axis=1) + np.random.normal(0, 0.5, samples)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Experiment data
experiment_data = {
    "input_feature_dimensionality_reduction": {
        "1D": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "3D": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "5D": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# Define a simple Bayesian linear regression model
class BayesianLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# Training and evaluation function
def train_and_evaluate(dim, X_train, X_val, y_train, y_val):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = BayesianLinearRegression(input_dim=dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        predictions = model(X_train_tensor)
        train_loss = criterion(predictions.squeeze(), y_train_tensor)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions.squeeze(), y_val_tensor)

        # Save metrics
        experiment_data["input_feature_dimensionality_reduction"][f"{dim}D"]["losses"][
            "train"
        ].append(train_loss.item())
        experiment_data["input_feature_dimensionality_reduction"][f"{dim}D"]["losses"][
            "val"
        ].append(val_loss.item())
        experiment_data["input_feature_dimensionality_reduction"][f"{dim}D"][
            "predictions"
        ].append(val_predictions.cpu().numpy())
        experiment_data["input_feature_dimensionality_reduction"][f"{dim}D"][
            "ground_truth"
        ].append(y_val)

        print(
            f"Dim: {dim}D, Epoch {epoch}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}"
        )


# Generate and run experiments for 1D, 3D, and 5D data
for dims in [1, 3, 5]:
    X_train, X_val, y_train, y_val = generate_synthetic_data(dimensions=dims)
    train_and_evaluate(dims, X_train, X_val, y_train, y_val)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
