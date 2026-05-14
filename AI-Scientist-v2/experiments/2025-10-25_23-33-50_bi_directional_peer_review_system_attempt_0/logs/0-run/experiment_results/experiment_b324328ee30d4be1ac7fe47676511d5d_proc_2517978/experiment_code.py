import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic dataset creation
np.random.seed(0)
num_samples = 1000
num_features = 5
X = np.random.rand(num_samples, num_features)  # feature representation of reviews
y = np.random.rand(num_samples)  # Reviewer Quality Index (RQI)

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# DataLoader preparation
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Simple neural network model
class RQIModel(nn.Module):
    def __init__(self):
        super(RQIModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


model = RQIModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation
experiment_data = {
    "synthetic_peer_review": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

for epoch in range(100):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in zip(["features", "labels"], batch)}
        optimizer.zero_grad()
        outputs = model(batch["features"])
        loss = criterion(outputs, batch["labels"].view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    experiment_data["synthetic_peer_review"]["losses"]["train"].append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in zip(["features", "labels"], batch)}
            outputs = model(batch["features"])
            loss = criterion(outputs, batch["labels"].view(-1, 1))
            val_loss += loss.item()
            experiment_data["synthetic_peer_review"]["predictions"].append(
                outputs.cpu().numpy()
            )
            experiment_data["synthetic_peer_review"]["ground_truth"].append(
                batch["labels"].cpu().numpy()
            )

    val_loss /= len(val_loader)
    experiment_data["synthetic_peer_review"]["losses"]["val"].append(val_loss)
    print(f"Epoch {epoch + 1}: validation_loss = {val_loss:.4f}")

# Evaluation metric (Average RQI)
experiment_data["synthetic_peer_review"]["metrics"]["train"].append(
    np.mean([r[0] for r in experiment_data["synthetic_peer_review"]["predictions"]])
)
experiment_data["synthetic_peer_review"]["metrics"]["val"].append(
    np.mean([r[0] for r in experiment_data["synthetic_peer_review"]["ground_truth"]])
)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
