import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Function to generate synthetic datasets with varying correlations
def generate_datasets(correlation_level):
    np.random.seed(0)
    size = 1000
    X = np.random.rand(size, 3)

    # Create correlated features
    if correlation_level == "low":
        X[:, 1] = X[:, 0] + (np.random.rand(size) - 0.5) * 0.2  # Low correlation
        X[:, 2] = X[:, 0] + (np.random.rand(size) - 0.5) * 0.2
    elif correlation_level == "medium":
        X[:, 1] = X[:, 0] + (np.random.rand(size) - 0.5) * 0.5  # Medium correlation
        X[:, 2] = X[:, 0] + (np.random.rand(size) - 0.5) * 0.5
    elif correlation_level == "high":
        X[:, 1] = X[:, 0] + (np.random.rand(size) - 0.5) * 0.9  # High correlation
        X[:, 2] = X[:, 0] + (np.random.rand(size) - 0.5) * 0.9

    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + np.random.normal(0, 0.1, size)

    return X, y


# Store experiment data
experiment_data = {"input_feature_correlation_analysis": {}}
correlation_levels = ["low", "medium", "high"]
batch_sizes = [16, 32, 64]
epochs = 100

for correlation_level in correlation_levels:
    X, y = generate_datasets(correlation_level)

    # Normalize Features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Define the model
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(3, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize storage for metrics
    experiment_data["input_feature_correlation_analysis"][correlation_level] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Training Loop for each batch size
    for batch_size in batch_sizes:
        model = SimpleNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create data loaders
        train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )

        # Training Loop
        for epoch in range(epochs):
            # Train
            model.train()
            for data in train_loader:
                inputs, targets = data
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                train_loss = criterion(outputs, targets)
                train_loss.backward()
                optimizer.step()

            experiment_data["input_feature_correlation_analysis"][correlation_level][
                "losses"
            ]["train"].append(train_loss.item())

            # Validate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor)
                experiment_data["input_feature_correlation_analysis"][
                    correlation_level
                ]["losses"]["val"].append(val_loss.item())

            # Calculate PWIS as a metric (for demonstration)
            PWIS = 1 - val_loss.item()  # Higher is better
            experiment_data["input_feature_correlation_analysis"][correlation_level][
                "metrics"
            ]["val"].append(PWIS)

            print(
                f"Correlation: {correlation_level}, Batch Size {batch_size}, Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss.item():.4f}, Validation Loss = {val_loss.item():.4f}, PWIS = {PWIS:.4f}"
            )

    # Store predictions and ground truth for future analysis
    experiment_data["input_feature_correlation_analysis"][correlation_level][
        "predictions"
    ].append(val_outputs.cpu().numpy())
    experiment_data["input_feature_correlation_analysis"][correlation_level][
        "ground_truth"
    ].append(y_val_tensor.cpu().numpy())

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
