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


# Define a simple neural network model with L2 regularization
class SimpleNNWithL2(nn.Module):
    def __init__(self):
        super(SimpleNNWithL2, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# Define a simple neural network model with Dropout
class SimpleNNWithDropout(nn.Module):
    def __init__(self):
        super(SimpleNNWithDropout, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))


# Initialize experiment data storage
experiment_data = {
    "L2_regularization": {
        "synthetic_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
    "dropout": {
        "synthetic_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
}


# Training function
def train_model(model, optimizer, criterion, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor).squeeze()
        train_loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()

        experiment_data[regularization_type]["synthetic_data"]["losses"][
            "train"
        ].append(train_loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)

        experiment_data[regularization_type]["synthetic_data"]["losses"]["val"].append(
            val_loss.item()
        )
        print(
            f"{regularization_type} - Epoch {epoch}: validation_loss = {val_loss:.4f}"
        )

        # Metrics calculation
        train_preds = (outputs > 0.5).float()
        val_preds = (val_outputs > 0.5).float()

        train_score = (train_preds == y_train_tensor).float().mean().item()
        val_score = (val_preds == y_val_tensor).float().mean().item()

        experiment_data[regularization_type]["synthetic_data"]["metrics"][
            "train"
        ].append(train_score)
        experiment_data[regularization_type]["synthetic_data"]["metrics"]["val"].append(
            val_score
        )


# Training with L2 Regularization
regularization_type = "L2_regularization"
model_l2 = SimpleNNWithL2().to(device)
criterion = nn.BCELoss()
optimizer_l2 = optim.Adam(
    model_l2.parameters(), lr=0.001, weight_decay=0.01
)  # L2 regularization
train_model(model_l2, optimizer_l2, criterion)

# Training with Dropout
regularization_type = "dropout"
model_dropout = SimpleNNWithDropout().to(device)
optimizer_dropout = optim.Adam(
    model_dropout.parameters(), lr=0.001
)  # No explicit weight decay
train_model(model_dropout, optimizer_dropout, criterion)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
