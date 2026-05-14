import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic Dataset Creation
np.random.seed(0)
x = np.random.uniform(-3, 3, (1000, 1))
y = np.sin(x) + np.random.normal(0, 0.1, (1000, 1))
x_tensor = torch.FloatTensor(x).to(device)
y_tensor = torch.FloatTensor(y).to(device)
dataset = TensorDataset(x_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model Definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Experiment Data Storage
experiment_data = {
    "synthetic_dataset": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training Loop
for epoch in range(50):  # simple training loop
    model.train()
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in zip(["x", "y"], batch)}
        optimizer.zero_grad()
        outputs = model(batch["x"])
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()

        experiment_data["synthetic_dataset"]["losses"]["train"].append(loss.item())

    # Validation step (using the same synthetic data for simplicity)
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_tensor)
        val_loss = criterion(val_outputs, y_tensor)
        experiment_data["synthetic_dataset"]["losses"]["val"].append(val_loss.item())

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

# Save predictions and ground truth
experiment_data["synthetic_dataset"]["predictions"] = val_outputs.cpu().numpy().tolist()
experiment_data["synthetic_dataset"]["ground_truth"] = y_tensor.cpu().numpy().tolist()

# Compute Reliability Measure
pred_intervals = 0.2  # user-specified risk preference for prediction intervals
reliable_count = (
    (val_outputs.cpu().numpy() >= (y_tensor.cpu().numpy() - pred_intervals))
    & (val_outputs.cpu().numpy() <= (y_tensor.cpu().numpy() + pred_intervals))
).sum()
reliability_measure = reliable_count / len(y_tensor)
experiment_data["synthetic_dataset"]["metrics"]["train"].append(reliability_measure)

# Save Experiment Data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
