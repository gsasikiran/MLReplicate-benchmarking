import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Create synthetic data
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert data to torch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)  # Changed to remove unsqueeze
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)  # Changed to remove unsqueeze


# Define a simple neural network model with batch normalization
class SimpleNNWithBN(nn.Module):
    def __init__(self):
        super(SimpleNNWithBN, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 1)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        return torch.sigmoid(self.bn2(self.fc2(x)))


# Initialize model, loss function, and optimizer
model = SimpleNNWithBN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize experiment data storage
experiment_data = {
    "hyperparam_tuning_batch_norm": {
        "synthetic_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Training loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor).squeeze()  # Outputs should be of shape [800]
    train_loss = criterion(outputs, y_train_tensor)  # y_train_tensor is of shape [800]

    # Backward pass and optimization
    train_loss.backward()
    optimizer.step()

    experiment_data["hyperparam_tuning_batch_norm"]["synthetic_data"]["losses"][
        "train"
    ].append(train_loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor).squeeze()
        val_loss = criterion(val_outputs, y_val_tensor)

    experiment_data["hyperparam_tuning_batch_norm"]["synthetic_data"]["losses"][
        "val"
    ].append(val_loss.item())
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

    # Simple metric calculation based on predictions
    train_preds = (outputs > 0.5).float()
    val_preds = (val_outputs > 0.5).float()

    train_score = (
        (train_preds == y_train_tensor).float().mean().item()
    )  # Adjusted for shape match
    val_score = (
        (val_preds == y_val_tensor).float().mean().item()
    )  # Adjusted for shape match

    experiment_data["hyperparam_tuning_batch_norm"]["synthetic_data"]["metrics"][
        "train"
    ].append(train_score)
    experiment_data["hyperparam_tuning_batch_norm"]["synthetic_data"]["metrics"][
        "val"
    ].append(val_score)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
