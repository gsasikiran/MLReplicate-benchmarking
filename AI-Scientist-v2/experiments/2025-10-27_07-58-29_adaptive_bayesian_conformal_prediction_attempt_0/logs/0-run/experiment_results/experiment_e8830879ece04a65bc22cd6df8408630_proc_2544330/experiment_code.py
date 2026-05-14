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
def generate_data(noise_level):
    np.random.seed(42)
    X = np.random.rand(1000, 1)
    y = 3 * X.squeeze() + np.random.normal(0, noise_level, X.shape[0])
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Experiment data
experiment_data = {"multiple_synthetic_datasets": {}}

# Noise levels
noise_levels = {"low": 0.1, "medium": 0.5, "high": 1.0}

# Loop over different noise levels
for noise_name, noise_level in noise_levels.items():
    print(f"\nGenerating data with {noise_name} noise...")
    X_train, X_val, y_train, y_val = generate_data(noise_level)

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Store experiment data for this noise level
    experiment_data["multiple_synthetic_datasets"][noise_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Train Bayesian Linear Regression model
    model = nn.Linear(1, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_train_tensor)
        train_loss = criterion(predictions.squeeze(), y_train_tensor)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions.squeeze(), y_val_tensor)

        # Save metrics and losses
        experiment_data["multiple_synthetic_datasets"][noise_name]["losses"][
            "train"
        ].append(train_loss.item())
        experiment_data["multiple_synthetic_datasets"][noise_name]["losses"][
            "val"
        ].append(val_loss.item())
        experiment_data["multiple_synthetic_datasets"][noise_name][
            "predictions"
        ].append(val_predictions.cpu().numpy())
        experiment_data["multiple_synthetic_datasets"][noise_name][
            "ground_truth"
        ].append(y_val)

        print(
            f"Noise Level: {noise_name} | Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

    # Reliability measure
    reliable_predictions = (val_predictions.cpu().numpy().flatten() >= 2.5) & (
        val_predictions.cpu().numpy().flatten() <= 3.5
    )
    reliability_measure = np.mean(reliable_predictions)
    experiment_data["multiple_synthetic_datasets"][noise_name]["metrics"]["val"].append(
        reliability_measure
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
