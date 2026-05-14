import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


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

# Hyperparameter tuning setup
epoch_list = [5, 10, 20, 30]  # Different values for num_epochs
experiment_data = {
    "early_stopping": {
        "synthetic_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

patience = 3  # How many epochs to wait before stopping
for num_epochs in epoch_list:
    model = SimpleNN().to(device)  # Reinitialize the model for each epoch
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in train_loader:
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

        train_loss = epoch_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        experiment_data["early_stopping"]["synthetic_dataset"]["losses"][
            "train"
        ].append(train_loss)
        experiment_data["early_stopping"]["synthetic_dataset"]["metrics"][
            "train"
        ].append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_outputs = model(val_X).squeeze()
                loss = criterion(val_outputs, val_y)

                val_loss += loss.item()
                val_predicted = (val_outputs > 0.5).float()
                val_correct += (val_predicted == val_y).sum().item()
                val_samples += val_y.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_samples
        experiment_data["early_stopping"]["synthetic_dataset"]["losses"]["val"].append(
            val_loss
        )
        experiment_data["early_stopping"]["synthetic_dataset"]["metrics"]["val"].append(
            val_accuracy
        )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(
            f"Epoch {epoch + 1}/{num_epochs}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}, val_loss = {val_loss:.4f}, val_accuracy = {val_accuracy:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
