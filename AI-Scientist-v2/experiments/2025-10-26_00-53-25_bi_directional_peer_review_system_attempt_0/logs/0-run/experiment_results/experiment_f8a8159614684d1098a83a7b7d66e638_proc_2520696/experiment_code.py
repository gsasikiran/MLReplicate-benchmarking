import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Function to generate synthetic dataset
def generate_synthetic_data(num_samples, params):
    X = np.random.rand(num_samples, 3) * params["scale"]
    RQS = np.clip(
        X[:, 0] * params["weight_clarity"]
        + X[:, 1] * params["weight_depth"]
        + X[:, 2] * params["weight_relevance"]
        + np.random.normal(0, params["noise"], num_samples),
        0,
        1,
    )
    return X, RQS


# Generate different synthetic datasets with varying parameters
datasets_params = [
    {
        "scale": 1,
        "weight_clarity": 0.5,
        "weight_depth": 0.3,
        "weight_relevance": 0.2,
        "noise": 0.05,
    },
    {
        "scale": 1.5,
        "weight_clarity": 0.6,
        "weight_depth": 0.2,
        "weight_relevance": 0.2,
        "noise": 0.1,
    },
    {
        "scale": 0.8,
        "weight_clarity": 0.4,
        "weight_depth": 0.4,
        "weight_relevance": 0.2,
        "noise": 0.02,
    },
]

# Store experiment data
experiment_data = {"multiple_synthetic_datasets": {}}

num_epochs = 50
learning_rates = [0.0001, 0.001, 0.01]

for idx, params in enumerate(datasets_params):
    print(f"\nGenerating dataset {idx+1} with parameters: {params}")
    num_samples = 1000
    X, RQS = generate_synthetic_data(num_samples, params)

    X_train, X_val, y_train, y_val = train_test_split(
        X, RQS, test_size=0.2, random_state=42
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Define the neural network model
    class RQSModel(nn.Module):
        def __init__(self):
            super(RQSModel, self).__init__()
            self.fc1 = nn.Linear(3, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

    # Hyperparameter tuning
    experiment_data["multiple_synthetic_datasets"][f"dataset_{idx+1}"] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for lr in learning_rates:
        print(f"\nTraining dataset {idx+1} with learning rate: {lr}")

        model = RQSModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            y_train_pred = model(X_train_tensor)
            train_loss = criterion(y_train_pred.squeeze(), y_train_tensor)
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_tensor)
                val_loss = criterion(y_val_pred.squeeze(), y_val_tensor)

            # Update metrics
            experiment_data["multiple_synthetic_datasets"][f"dataset_{idx+1}"][
                "metrics"
            ]["train"].append(1 - train_loss.item())
            experiment_data["multiple_synthetic_datasets"][f"dataset_{idx+1}"][
                "losses"
            ]["train"].append(train_loss.item())
            experiment_data["multiple_synthetic_datasets"][f"dataset_{idx+1}"][
                "metrics"
            ]["val"].append(1 - val_loss.item())
            experiment_data["multiple_synthetic_datasets"][f"dataset_{idx+1}"][
                "losses"
            ]["val"].append(val_loss.item())
            experiment_data["multiple_synthetic_datasets"][f"dataset_{idx+1}"][
                "predictions"
            ].append(y_val_pred.cpu().numpy())
            experiment_data["multiple_synthetic_datasets"][f"dataset_{idx+1}"][
                "ground_truth"
            ].append(y_val_tensor.cpu().numpy())

            print(
                f"Epoch {epoch+1}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
            )

# Save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
