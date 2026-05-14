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


# Experiment data
experiment_data = {
    "optimizer_comparison": {
        "SGD": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "Adam": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "RMSprop": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "Adagrad": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    "Adagrad": optim.Adagrad,
}

# Training loop for different optimizers
for optimizer_name, optimizer_class in optimizers.items():
    print(f"Training with optimizer: {optimizer_name}")

    model = BayesianLinearRegression().to(device)
    criterion = nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), lr=0.01)

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
        experiment_data["optimizer_comparison"][optimizer_name]["losses"][
            "train"
        ].append(train_loss.item())
        experiment_data["optimizer_comparison"][optimizer_name]["losses"]["val"].append(
            val_loss.item()
        )
        experiment_data["optimizer_comparison"][optimizer_name]["predictions"].append(
            val_predictions.cpu().numpy()
        )
        experiment_data["optimizer_comparison"][optimizer_name]["ground_truth"].append(
            y_val
        )

        print(
            f"Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

# Reliability measure for one of the optimizers as a demonstration
for optimizer_name in optimizers.keys():
    val_predictions_flat = np.concatenate(
        experiment_data["optimizer_comparison"][optimizer_name]["predictions"]
    ).flatten()
    reliable_predictions = (val_predictions_flat >= 2.5) & (val_predictions_flat <= 3.5)
    reliability_measure = np.mean(reliable_predictions)
    experiment_data["optimizer_comparison"][optimizer_name]["metrics"]["val"].append(
        reliability_measure
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
