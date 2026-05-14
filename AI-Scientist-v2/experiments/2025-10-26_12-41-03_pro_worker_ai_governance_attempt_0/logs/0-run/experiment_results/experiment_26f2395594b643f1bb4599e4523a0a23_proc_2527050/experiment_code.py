import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Create synthetic dataset
np.random.seed(42)
num_samples = 1000
job_displacement = np.random.rand(num_samples)
wage_change = np.random.rand(num_samples)
worker_satisfaction = np.random.rand(num_samples)
WIS = (
    0.5 * job_displacement + 0.3 * wage_change + 0.2 * worker_satisfaction
)  # Simplified relationship

X = np.column_stack((job_displacement, wage_change, worker_satisfaction))
y = WIS

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


# Instantiate model and optimizer
model = SimpleModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
num_epochs = 100
experiment_data = {
    "synthetic_dataset": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

for epoch in range(num_epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    train_loss = criterion(outputs.squeeze(), y_train_tensor)
    train_loss.backward()
    optimizer.step()

    experiment_data["synthetic_dataset"]["losses"]["train"].append(train_loss.item())

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
        experiment_data["synthetic_dataset"]["losses"]["val"].append(val_loss.item())

        WIS_val = val_outputs.squeeze().cpu().numpy()
        experiment_data["synthetic_dataset"]["metrics"]["val"].append(np.mean(WIS_val))

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
