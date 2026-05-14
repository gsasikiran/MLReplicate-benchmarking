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

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 1)
y = 3 * X.squeeze() + np.random.normal(0, 0.5, X.shape[0])

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define a simple Bayesian linear regression model
class BayesianLinearRegression(nn.Module):
    def __init__(self, l1_lambda=0.0, l2_lambda=0.0):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        l1_loss = self.l1_lambda * torch.sum(torch.abs(self.linear.weight))
        l2_loss = self.l2_lambda * torch.sum(self.linear.weight**2)
        return l1_loss + l2_loss


# Experiment data
experiment_data = {
    "no_reg": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "l1_reg": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "l2_reg": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}


# ACIW metric storage
def calculate_aciw(predictions, ground_truth):
    return np.mean(np.abs(predictions - ground_truth))


# Training function
def train_model(
    model, key, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs=100
):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()

        # Forward pass
        predictions = model(X_train_tensor)
        train_loss = (
            criterion(predictions.squeeze(), y_train_tensor)
            + model.regularization_loss()
        )

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = (
                criterion(val_predictions.squeeze(), y_val_tensor)
                + model.regularization_loss()
            )

        # Save metrics
        experiment_data[key]["losses"]["train"].append(train_loss.item())
        experiment_data[key]["losses"]["val"].append(val_loss.item())
        experiment_data[key]["predictions"].append(val_predictions.cpu().numpy())
        experiment_data[key]["ground_truth"].append(y_val)

        # Calculate ACIW
        a_ci_width = calculate_aciw(val_predictions.cpu().numpy(), y_val)
        experiment_data[key]["metrics"]["val"].append(a_ci_width)

        print(
            f"{key}: Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, ACIW = {a_ci_width:.4f}"
        )


# Training configurations
regularization_configs = {
    "no_reg": (0.0, 0.0),
    "l1_reg": (0.1, 0.0),  # L1 regularization
    "l2_reg": (0.0, 0.1),  # L2 regularization
}

# Train models
for key, (l1_lambda, l2_lambda) in regularization_configs.items():
    print(f"Training with {key}...")
    model = BayesianLinearRegression(l1_lambda=l1_lambda, l2_lambda=l2_lambda).to(
        device
    )
    train_model(model, key, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
