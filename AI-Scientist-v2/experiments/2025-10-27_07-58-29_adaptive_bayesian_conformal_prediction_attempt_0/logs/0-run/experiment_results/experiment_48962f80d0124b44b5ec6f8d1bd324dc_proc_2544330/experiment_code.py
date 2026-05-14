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


# Generate multiple synthetic datasets with varying noise
def generate_synthetic_data(noise_type, noise_params, size=1000):
    X = np.random.rand(size, 1)
    if noise_type == "gaussian":
        y = 3 * X.squeeze() + np.random.normal(
            noise_params[0], noise_params[1], X.shape[0]
        )
    elif noise_type == "uniform":
        y = 3 * X.squeeze() + np.random.uniform(
            noise_params[0], noise_params[1], X.shape[0]
        )
    else:
        raise ValueError("Unsupported noise type")
    return X, y


datasets = {
    "Gaussian_Low_Noise": generate_synthetic_data("gaussian", (0, 0.5)),
    "Uniform_Noise": generate_synthetic_data("uniform", (-0.5, 0.5)),
    "Gaussian_High_Noise": generate_synthetic_data("gaussian", (0, 2.0)),
}

experiment_data = {}


# Define the model
class BayesianLinearRegression(nn.Module):
    def __init__(self):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Hyperparameter tuning for momentum
momentum_values = [0.0, 0.5, 0.9]
for dataset_name, (X, y) in datasets.items():
    experiment_data[dataset_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    for momentum in momentum_values:
        print(f"Training on {dataset_name} with momentum: {momentum}")

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
            experiment_data[dataset_name]["losses"]["train"].append(train_loss.item())
            experiment_data[dataset_name]["losses"]["val"].append(val_loss.item())
            experiment_data[dataset_name]["predictions"].append(
                val_predictions.cpu().numpy()
            )
            experiment_data[dataset_name]["ground_truth"].append(y_val)

            print(
                f"[{dataset_name}] Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
            )

        # Reliability measure
        reliable_predictions = (val_predictions.cpu().numpy().flatten() >= 2.5) & (
            val_predictions.cpu().numpy().flatten() <= 3.5
        )
        reliability_measure = np.mean(reliable_predictions)
        experiment_data[dataset_name]["metrics"]["val"].append(reliability_measure)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
