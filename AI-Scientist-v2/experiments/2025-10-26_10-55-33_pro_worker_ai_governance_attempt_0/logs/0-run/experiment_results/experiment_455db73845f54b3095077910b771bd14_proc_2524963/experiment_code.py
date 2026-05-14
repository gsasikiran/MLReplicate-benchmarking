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


# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)  # Use dynamic input size
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Ablation study configuration
feature_names = ["job displacement rate", "income stability", "empowerment score"]
ablation_data = {"feature_importance": {}}

# Training Loop for each feature removal case
epochs = 100
for feature_idx, feature_name in enumerate(feature_names):
    ablation_data["feature_importance"][feature_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Remove the feature by masking
    X_train_ablation = np.delete(X_train, feature_idx, axis=1)
    X_val_ablation = np.delete(X_val, feature_idx, axis=1)

    # Convert to PyTorch tensors
    X_train_tensor_ablation = torch.tensor(X_train_ablation, dtype=torch.float32).to(
        device
    )
    y_train_tensor_ablation = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor_ablation = torch.tensor(X_val_ablation, dtype=torch.float32).to(device)
    y_val_tensor_ablation = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = SimpleNN(input_size=X_train_ablation.shape[1]).to(
        device
    )  # Pass new input size
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create data loaders
    train_data = torch.utils.data.TensorDataset(
        X_train_tensor_ablation, y_train_tensor_ablation
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # Training Loop
    for epoch in range(epochs):
        # Train
        model.train()
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(
                device
            )  # Move inputs and targets to device
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            train_loss = criterion(outputs, targets)
            train_loss.backward()
            optimizer.step()

        ablation_data["feature_importance"][feature_name]["losses"]["train"].append(
            train_loss.item()
        )

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor_ablation).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor_ablation)
            ablation_data["feature_importance"][feature_name]["losses"]["val"].append(
                val_loss.item()
            )

        # Calculate PWIS as a metric
        PWIS = 1 - val_loss.item()  # Higher is better
        ablation_data["feature_importance"][feature_name]["metrics"]["val"].append(PWIS)

        print(
            f"Removed Feature: {feature_name}, Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss.item():.4f}, Validation Loss = {val_loss.item():.4f}, PWIS = {PWIS:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), ablation_data)
