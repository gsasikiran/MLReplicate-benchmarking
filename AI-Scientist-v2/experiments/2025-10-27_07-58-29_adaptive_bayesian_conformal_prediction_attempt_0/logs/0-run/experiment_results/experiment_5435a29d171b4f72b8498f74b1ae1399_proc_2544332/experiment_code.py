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


# Define Bayesian Linear Regression model
class BayesianLinearRegression(nn.Module):
    def __init__(self):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Function to generate synthetic data with varying noise levels
def generate_data(noise_level, n_samples=1000):
    np.random.seed(42)
    X = np.random.rand(n_samples, 1)
    y = 3 * X.squeeze() + np.random.normal(0, noise_level, n_samples)
    return X, y


# Initialize experiment data structure
experiment_data = {"noise_level_impact": {}}

# Different noise levels
noise_levels = [0.1, 0.5, 1.0]
for noise in noise_levels:
    print(f"Processing dataset with noise level: {noise}")

    # Generate data
    X, y = generate_data(noise)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Initialize model, criterion and optimizer
    model = BayesianLinearRegression().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.5
    )  # Fixed momentum for comparison

    # Store metrics for this noise level
    experiment_data["noise_level_impact"][f"noise_{noise}"] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

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
        experiment_data["noise_level_impact"][f"noise_{noise}"]["losses"][
            "train"
        ].append(train_loss.item())
        experiment_data["noise_level_impact"][f"noise_{noise}"]["losses"]["val"].append(
            val_loss.item()
        )
        experiment_data["noise_level_impact"][f"noise_{noise}"]["predictions"].append(
            val_predictions.cpu().numpy()
        )
        experiment_data["noise_level_impact"][f"noise_{noise}"]["ground_truth"].append(
            y_val
        )

        # Compute Adaptive Confidence Interval Width (ACIW)
        ci_width = (val_predictions.max() - val_predictions.min()).item()
        experiment_data["noise_level_impact"][f"noise_{noise}"]["metrics"][
            "val"
        ].append(ci_width)

        print(
            f"Noise: {noise}, Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, ACIW = {ci_width:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
