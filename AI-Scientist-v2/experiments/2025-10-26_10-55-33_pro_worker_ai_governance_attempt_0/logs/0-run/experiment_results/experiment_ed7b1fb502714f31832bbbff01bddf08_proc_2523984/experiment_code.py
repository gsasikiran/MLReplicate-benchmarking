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

# Synthetic Data Generation
np.random.seed(0)
X = np.random.rand(
    1000, 3
)  # Features: job displacement rate, income stability, empowerment score
y = (
    0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + np.random.normal(0, 0.1, 1000)
)  # PWIS

# Normalize Features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# Experiment data storage
experiment_data = {
    "hyperparam_tuning_hidden_layer_size": {
        "synthetic_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}


# Define the model with variable hidden layer size
class SimpleNN(nn.Module):
    def __init__(self, hidden_layer_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Hyperparameter Tuning for hidden layer size
hidden_layer_sizes = [5, 10, 15, 20]  # Different sizes to test

# Training Loop
epochs = 100
for hidden_layer_size in hidden_layer_sizes:
    print(f"Training model with hidden layer size: {hidden_layer_size}")
    model = SimpleNN(hidden_layer_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        train_loss = criterion(outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_dataset"][
            "losses"
        ]["train"].append(train_loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_dataset"][
                "losses"
            ]["val"].append(val_loss.item())

        # Calculate PWIS as a metric (for demonstration)
        PWIS = 1 - val_loss.item()  # Higher is better
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_dataset"][
            "metrics"
        ]["val"].append(PWIS)

        print(
            f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss.item():.4f}, Validation Loss = {val_loss.item():.4f}, PWIS = {PWIS:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
