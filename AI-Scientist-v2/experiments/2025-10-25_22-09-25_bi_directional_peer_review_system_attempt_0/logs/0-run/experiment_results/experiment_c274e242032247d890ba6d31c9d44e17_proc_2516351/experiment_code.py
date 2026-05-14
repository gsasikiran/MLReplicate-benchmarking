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
features = np.random.rand(num_samples, 3)
labels = np.random.rand(num_samples)

# Prepare DataLoader
X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor, y_tensor)

# Split into train and validation datasets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Initialize model, loss, and optimizer
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare data storage
experiment_data = {
    "hyperparam_tuning_type_1": {
        "synthetic_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Early stopping parameters
patience = 5
best_val_loss = float("inf")
counter = 0

# Training loop
num_epochs = 100
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

    avg_train_loss = epoch_loss / len(train_loader)
    experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["losses"][
        "train"
    ].append(avg_train_loss)

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = {
                k: v.to(device) for k, v in zip(("features", "labels"), val_batch)
            }
            val_outputs = model(val_batch["features"])
            val_loss += criterion(val_outputs.squeeze(), val_batch["labels"]).item()

    avg_val_loss = val_loss / len(val_loader)
    experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["losses"][
        "val"
    ].append(avg_val_loss)

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
    else:
        counter += 1

    # Save training metrics
    rqs = 1 - avg_train_loss
    experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["metrics"][
        "train"
    ].append(rqs)

    print(
        f"Epoch {epoch+1}: train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}, RQS = {rqs:.4f}"
    )

    if counter >= patience:
        print("Early stopping triggered.")
        break

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
