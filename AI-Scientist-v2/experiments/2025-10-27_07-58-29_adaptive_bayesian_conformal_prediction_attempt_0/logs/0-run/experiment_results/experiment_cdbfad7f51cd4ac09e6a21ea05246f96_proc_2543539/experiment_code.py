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


# Define a Bayesian linear regression model with variable layers
class BayesianLinearRegression(nn.Module):
    def __init__(self, num_layers):
        super(BayesianLinearRegression, self).__init__()
        layers = []
        input_dim = 1
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, 10))  # 10 hidden units
            layers.append(nn.ReLU())
            input_dim = 10
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Data for experiment
experiment_data = {"hyperparam_tuning_num_layers": {}}

# Hyperparameter tuning for number of layers
for num_layers in range(1, 6):  # Test from 1 to 5 layers
    model = BayesianLinearRegression(num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    experiment_data["hyperparam_tuning_num_layers"][f"{num_layers}_layers"] = {
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
        experiment_data["hyperparam_tuning_num_layers"][f"{num_layers}_layers"][
            "losses"
        ]["train"].append(train_loss.item())
        experiment_data["hyperparam_tuning_num_layers"][f"{num_layers}_layers"][
            "losses"
        ]["val"].append(val_loss.item())
        experiment_data["hyperparam_tuning_num_layers"][f"{num_layers}_layers"][
            "predictions"
        ].append(val_predictions.cpu().numpy())
        experiment_data["hyperparam_tuning_num_layers"][f"{num_layers}_layers"][
            "ground_truth"
        ].append(y_val)

        print(
            f"Layers {num_layers}, Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
