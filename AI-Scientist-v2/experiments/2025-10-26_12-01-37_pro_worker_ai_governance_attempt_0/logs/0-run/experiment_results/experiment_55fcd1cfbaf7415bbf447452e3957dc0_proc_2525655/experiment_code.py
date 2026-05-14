import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic data generation
np.random.seed(42)
num_samples = 1000
features = np.random.rand(num_samples, 5)  # 5 features
labels = features @ np.array([0.3, -0.5, 1.0, 0.2, -0.1]) + np.random.normal(
    0, 0.1, num_samples
)  # Linear relation with noise
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Neural network definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

experiment_data = {
    "synthetic_data": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    experiment_data["synthetic_data"]["losses"]["train"].append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_targets = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_targets)
        experiment_data["synthetic_data"]["losses"]["val"].append(val_loss.item())

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

# Save all metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Visualization of predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_val, val_outputs.cpu().numpy(), alpha=0.5)
plt.xlabel("Ground Truth EIS")
plt.ylabel("Predicted EIS")
plt.title("Ground Truth vs Predicted EIS")
plt.savefig(os.path.join(working_dir, "predicted_vs_ground_truth.png"))
