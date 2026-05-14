import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set up working directory for saving data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Synthetic data generation
def generate_synthetic_data(num_samples):
    np.random.seed(42)
    job_retention = np.random.rand(num_samples)
    worker_satisfaction = np.random.rand(num_samples)
    fairness_distribution = np.random.rand(num_samples)
    return np.column_stack((job_retention, worker_satisfaction, fairness_distribution))


# Pro-Worker Impact Score calculation
def calculate_pwis(features):
    return np.mean(features, axis=1)  # Simple average for demonstration


# Neural Network Model
class PWISModel(nn.Module):
    def __init__(self):
        super(PWISModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generating Data
num_samples = 1000
features = generate_synthetic_data(num_samples)
pwis = calculate_pwis(features).reshape(-1, 1)

# Creating Dataset and DataLoader
X = torch.FloatTensor(features).to(device)
y = torch.FloatTensor(pwis).to(device)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss, Optimizer
model = PWISModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment Data Storage
experiment_data = {
    "pwis_experiment": {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(dataloader)
    experiment_data["pwis_experiment"]["losses"]["train"].append(avg_train_loss)

    # Calculate and store PWIS
    pwis_predictions = model(X).detach().cpu().numpy()
    experiment_data["pwis_experiment"]["predictions"].append(pwis_predictions.mean())
    experiment_data["pwis_experiment"]["ground_truth"].append(pwis.mean())

    print(f"Epoch {epoch + 1}: training_loss = {avg_train_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
