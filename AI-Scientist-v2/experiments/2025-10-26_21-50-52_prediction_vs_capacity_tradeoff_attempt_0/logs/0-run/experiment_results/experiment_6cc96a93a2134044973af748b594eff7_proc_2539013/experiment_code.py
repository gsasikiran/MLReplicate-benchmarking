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


# Model definition with Dropout
class SimpleNN(nn.Module):
    def __init__(self, dropout_rate):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Applying dropout
        return torch.sigmoid(self.fc2(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter tuning setup
dropout_rates = [0.0, 0.2, 0.5]
num_epochs = 20  # Fixed number of epochs for comparison
experiment_data = {
    "dropout_variation": {
        "synthetic_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

for dropout_rate in dropout_rates:
    model = SimpleNN(dropout_rate).to(
        device
    )  # Reinitialize the model for each dropout rate
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        experiment_data["dropout_variation"]["synthetic_dataset"]["losses"][
            "train"
        ].append(train_loss)
        experiment_data["dropout_variation"]["synthetic_dataset"]["metrics"][
            "train"
        ].append(train_accuracy)

        # Print metrics
        print(
            f"Dropout rate {dropout_rate} - Epoch {epoch + 1}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
