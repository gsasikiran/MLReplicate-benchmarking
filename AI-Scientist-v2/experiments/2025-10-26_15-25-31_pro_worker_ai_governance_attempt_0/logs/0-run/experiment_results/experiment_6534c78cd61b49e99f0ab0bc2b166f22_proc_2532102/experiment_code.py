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
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)


# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# Initialize experiment data storage
experiment_data = {
    "batch_size_ablation": {
        "synthetic_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "batch_sizes": [],
        }
    }
}

# List of batch sizes to test
batch_sizes = [16, 32, 64]
learning_rate = 0.001  # fixed learning rate for simplicity

for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    model = SimpleNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with batch processing
    for epoch in range(50):
        model.train()
        epoch_train_loss = 0
        num_batches = len(X_train_tensor) // batch_size + (
            len(X_train_tensor) % batch_size != 0
        )

        for i in range(num_batches):
            batch_X = X_train_tensor[i * batch_size : (i + 1) * batch_size]
            batch_y = y_train_tensor[i * batch_size : (i + 1) * batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            train_loss = criterion(outputs, batch_y)
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()

        avg_train_loss = epoch_train_loss / num_batches
        experiment_data["batch_size_ablation"]["synthetic_data"]["losses"][
            "train"
        ].append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)

        experiment_data["batch_size_ablation"]["synthetic_data"]["losses"][
            "val"
        ].append(val_loss.item())
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

        # Simple metric calculation based on predictions
        train_preds = (outputs > 0.5).float()
        val_preds = (val_outputs > 0.5).float()

        train_score = (train_preds == batch_y).float().mean().item()
        val_score = (val_preds == y_val_tensor).float().mean().item()

        experiment_data["batch_size_ablation"]["synthetic_data"]["metrics"][
            "train"
        ].append(train_score)
        experiment_data["batch_size_ablation"]["synthetic_data"]["metrics"][
            "val"
        ].append(val_score)

    experiment_data["batch_size_ablation"]["synthetic_data"]["batch_sizes"].append(
        batch_size
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
