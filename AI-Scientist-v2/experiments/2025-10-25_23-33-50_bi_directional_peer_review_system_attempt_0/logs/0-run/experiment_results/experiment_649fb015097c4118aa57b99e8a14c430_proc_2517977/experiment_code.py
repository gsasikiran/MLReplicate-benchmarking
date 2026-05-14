import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegressionModel(input_dim=10).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
experiment_data = {
    "synthetic_data": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
        optimizer.zero_grad()
        outputs = model(batch["X"]).squeeze()
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    experiment_data["synthetic_data"]["losses"]["train"].append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
            outputs = model(batch["X"]).squeeze()
            loss = criterion(outputs, batch["y"])
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    experiment_data["synthetic_data"]["losses"]["val"].append(avg_val_loss)
    print(f"Epoch {epoch+1}: validation_loss = {avg_val_loss:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
