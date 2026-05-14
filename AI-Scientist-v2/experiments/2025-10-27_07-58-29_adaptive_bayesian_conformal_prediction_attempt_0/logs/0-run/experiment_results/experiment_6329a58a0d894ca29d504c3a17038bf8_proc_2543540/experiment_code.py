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
    def __init__(self):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Hyperparameter tuning configuration for L2 regularization strength
l2_lambda_values = [0, 0.01, 0.1, 1.0, 10.0]  # Different values for L2 regularization
experiment_data = {"hyperparam_tuning_L2": {}}

for l2_lambda in l2_lambda_values:
    # Initialize model, loss function, and optimizer
    model = BayesianLinearRegression().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Store metrics and losses
    experiment_data["hyperparam_tuning_L2"][l2_lambda] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Training loop
    for epoch in range(100):
        model.train()

        # Forward pass
        predictions = model(X_train_tensor)
        train_loss = criterion(predictions.squeeze(), y_train_tensor)

        # Add L2 regularization
        l2_reg = sum(param.norm(2) for param in model.parameters())
        train_loss += l2_lambda * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions.squeeze(), y_val_tensor)

            # Add L2 regularization to validation loss (not needed for evaluation but may be of interest)
            val_loss += l2_lambda * l2_reg

        # Save metrics
        experiment_data["hyperparam_tuning_L2"][l2_lambda]["losses"]["train"].append(
            train_loss.item()
        )
        experiment_data["hyperparam_tuning_L2"][l2_lambda]["losses"]["val"].append(
            val_loss.item()
        )
        experiment_data["hyperparam_tuning_L2"][l2_lambda]["predictions"].append(
            val_predictions.cpu().numpy()
        )
        experiment_data["hyperparam_tuning_L2"][l2_lambda]["ground_truth"].append(y_val)

        print(
            f"L2_lambda {l2_lambda}, Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

# Reliability measure demonstration
for l2_lambda in l2_lambda_values:
    val_predictions_flat = np.concatenate(
        experiment_data["hyperparam_tuning_L2"][l2_lambda]["predictions"]
    ).flatten()
    reliable_predictions = (val_predictions_flat >= 2.5) & (val_predictions_flat <= 3.5)
    reliability_measure = np.mean(reliable_predictions)
    experiment_data["hyperparam_tuning_L2"][l2_lambda]["metrics"]["val"].append(
        reliability_measure
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
