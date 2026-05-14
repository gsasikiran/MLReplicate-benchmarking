import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Generate synthetic dataset
class WorkerDataset(Dataset):
    def __init__(self, size):
        np.random.seed(42)
        self.data = np.random.rand(
            size, 4
        )  # Features: Job Satisfaction, Displacement, Retraining Opportunities, Benefit Distribution
        self.labels = (
            self.data[:, 0] * 0.4
            + (1 - self.data[:, 1]) * 0.3
            + self.data[:, 2] * 0.2
            + self.data[:, 3] * 0.1
        )  # WWBI computation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32).to(
            device
        ), torch.tensor(self.labels[idx], dtype=torch.float32).to(device)


# Model definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Training function
def train_model(dataset, epochs=100):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    experiment_data = {
        "dataset_name": {"metrics": {"train": []}, "losses": {"train": []}}
    }

    for epoch in range(epochs):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Track metrics
        experiment_data["dataset_name"]["losses"]["train"].append(loss.item())
        train_metric = -loss.item()  # Dummy metric for WWBI, to be maximized
        experiment_data["dataset_name"]["metrics"]["train"].append(train_metric)
        print(f"Epoch {epoch + 1}: training_loss = {loss.item():.4f}")

    # Save metrics
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# Running the experiment
dataset = WorkerDataset(size=1000)
train_model(dataset)
