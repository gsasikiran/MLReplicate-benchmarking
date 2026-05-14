import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic data generation
np.random.seed(42)
num_samples = 1000
num_features = 10
X = np.random.randn(num_samples, num_features).astype(np.float32)
y = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome

# Create Dataset and DataLoader
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hyperparameters
num_epochs = 50
patience = 5  # Number of epochs to wait for improvement
best_val_accuracy = 0
epochs_without_improvement = 0

# Experiment data storage
experiment_data = {
    "hyperparam_tuning_early_stopping": {
        "synthetic_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Training loop with validation
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_X, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total_correct += (predicted == batch_y).sum().item()
        total_samples += batch_y.size(0)

    train_loss = epoch_loss / len(data_loader)
    train_accuracy = total_correct / total_samples
    experiment_data["hyperparam_tuning_early_stopping"]["synthetic_dataset"]["losses"][
        "train"
    ].append(train_loss)
    experiment_data["hyperparam_tuning_early_stopping"]["synthetic_dataset"]["metrics"][
        "train"
    ].append(train_accuracy)

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(batch_X).squeeze()
        val_loss = criterion(val_outputs, batch_y)
        val_accuracy = (val_outputs > 0.5).float().eq(
            batch_y
        ).sum().item() / batch_y.size(0)

    experiment_data["hyperparam_tuning_early_stopping"]["synthetic_dataset"]["losses"][
        "val"
    ].append(val_loss.item())
    experiment_data["hyperparam_tuning_early_stopping"]["synthetic_dataset"]["metrics"][
        "val"
    ].append(val_accuracy)

    # Early stopping check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping on epoch {epoch + 1}")
        break

    print(
        f"Epoch {epoch + 1}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}, val_loss = {val_loss.item():.4f}, val_accuracy = {val_accuracy:.4f}"
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
