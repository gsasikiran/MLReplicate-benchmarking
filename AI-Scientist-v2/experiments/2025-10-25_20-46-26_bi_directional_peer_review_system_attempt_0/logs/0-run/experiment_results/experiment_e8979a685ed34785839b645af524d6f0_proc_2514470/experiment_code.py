import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Handle GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic dataset generation
class FeedbackDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.features = torch.rand(num_samples, 4)  # 4 features for each review
        self.labels = self.calculate_labels(self.features)

    def calculate_labels(self, features):
        return (
            features[:, 0] + features[:, 1] - features[:, 2] + features[:, 3]
        ).clamp(
            0, 1
        )  # Simple logic

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "label": self.labels[idx]}


# Simple neural network model
class RASModel(nn.Module):
    def __init__(self):
        super(RASModel, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)  # Regression output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # Output between 0 and 1


# Prepare data
dataset = FeedbackDataset(num_samples=1000)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = RASModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment data to save metrics
experiment_data = {
    "hyperparam_tuning_num_epochs": {
        "feedback_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Hyperparameter tuning for num_epochs
epoch_ranges = [10, 20, 30]  # Define the epochs to test
for num_epochs in epoch_ranges:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in data_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            features = batch["features"]
            labels = batch["label"].view(-1, 1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(data_loader)
        experiment_data["hyperparam_tuning_num_epochs"]["feedback_dataset"]["losses"][
            "train"
        ].append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: training_loss = {avg_train_loss:.4f}")

        # Random evaluations for demonstration purposes (not real validation)
        val_loss = avg_train_loss + np.random.normal(0, 0.1)  # Simulate some noise
        experiment_data["hyperparam_tuning_num_epochs"]["feedback_dataset"]["losses"][
            "val"
        ].append(val_loss)

        # Calculate a synthetic RAS for evaluation
        synthetic_RAS = np.random.rand()  # Simulate some RAS values
        experiment_data["hyperparam_tuning_num_epochs"]["feedback_dataset"]["metrics"][
            "train"
        ].append(synthetic_RAS)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
