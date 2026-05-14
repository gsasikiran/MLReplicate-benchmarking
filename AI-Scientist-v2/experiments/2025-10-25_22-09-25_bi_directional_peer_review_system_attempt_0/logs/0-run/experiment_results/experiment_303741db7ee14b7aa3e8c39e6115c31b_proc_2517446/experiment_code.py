import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Handle GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic dataset
np.random.seed(0)
num_samples = 1000
features = np.random.rand(
    num_samples, 3
)  # e.g., thoroughness, constructiveness, acceptance rate
labels = np.random.rand(num_samples)  # RQS values

# Normalize features
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# Prepare DataLoader
X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define simple feedforward neural network with various activation functions
class SimpleNN(nn.Module):
    def __init__(self, hidden_units, activation_function):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        return self.fc2(x)


# Prepare data storage for ablation
experiment_data = {
    "impact_of_activation_functions": {
        "synthetic_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Activation functions to test
activation_functions = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Leaky ReLU": nn.LeakyReLU(0.01),
}

# Hyperparameter tuning for some configurations
hidden_units_list = [16, 32]
num_epochs = 10

for activation_name, activation_function in activation_functions.items():
    for hidden_units in hidden_units_list:
        print(
            f"\nTraining with activation: {activation_name}, hidden units: {hidden_units}"
        )
        # Initialize model with the specified activation function and hidden units
        model = SimpleNN(hidden_units, activation_function).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in zip(("features", "labels"), batch)}
                optimizer.zero_grad()
                outputs = model(batch["features"])
                loss = criterion(outputs.squeeze(), batch["labels"])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            rqs = 1 - avg_loss  # Placeholder for actual RQS calculation

            experiment_data["impact_of_activation_functions"]["synthetic_data"][
                "losses"
            ]["train"].append(avg_loss)
            experiment_data["impact_of_activation_functions"]["synthetic_data"][
                "metrics"
            ]["train"].append(rqs)

            print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}, RQS = {rqs:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
