import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
features = np.random.rand(n_samples, 4)  # 4 synthetic features
wbi = (features @ np.array([0.3, -0.5, 0.1, 0.2])) + 0.5  # WWBI computation
wbi = np.clip(wbi, 0, 1)  # WWBI must be between 0 and 1

# Create data loaders
X_train, X_val = features[:800], features[800:]
y_train, y_val = wbi[:800], wbi[800:]

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32).to(device),
    torch.tensor(y_train, dtype=torch.float32).to(device),
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32).to(device),
    torch.tensor(y_val, dtype=torch.float32).to(device),
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Model, loss function, optimizer
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation
experiment_data = {
    "synthetic_data": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        x_batch, y_batch = batch
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.flatten(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    experiment_data["synthetic_data"]["losses"]["train"].append(train_loss)

    model.eval()
    val_loss = 0
    predictions = []
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch
            outputs = model(x_batch)
            loss = criterion(outputs.flatten(), y_batch)
            val_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())

    val_loss /= len(val_loader)
    experiment_data["synthetic_data"]["losses"]["val"].append(val_loss)
    experiment_data["synthetic_data"]["predictions"].extend(predictions)
    experiment_data["synthetic_data"]["ground_truth"].extend(y_val)

    print(
        f"Epoch {epoch+1}: training_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
