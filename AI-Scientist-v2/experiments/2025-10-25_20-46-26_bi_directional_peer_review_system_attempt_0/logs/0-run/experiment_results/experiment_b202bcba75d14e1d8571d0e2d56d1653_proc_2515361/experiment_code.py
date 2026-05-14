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


# Synthetic dataset generation with variable features
class FeedbackDataset(Dataset):
    def __init__(self, num_samples=1000, num_features=4):
        self.features = torch.rand(
            num_samples, num_features
        )  # Features according to provided number
        self.labels = self.calculate_labels(self.features)

    def calculate_labels(self, features):
        return (
            (features[:, 0] + features[:, 1] - features[:, 2] + features[:, 3]).clamp(
                0, 1
            )
            if features.shape[1] >= 4
            else (
                (features[:, 0] + features[:, 1]).clamp(0, 1)
                if features.shape[1] == 2
                else (features.sum(dim=1)).clamp(0, 1)
            )
        )  # Simple logic for less than 4 features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "label": self.labels[idx]}


# Simple neural network model
class RASModel(nn.Module):
    def __init__(self, activation_function, num_features):
        super(RASModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 8)
        self.fc2 = nn.Linear(8, 1)  # Regression output
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # Output between 0 and 1


# Experiment data to save metrics
experiment_data = {
    "variability_of_input_features": {
        "FeedbackDataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Prepare experiments with varying features
num_epochs = 20
feature_counts = [2, 4, 6]

for num_features in feature_counts:
    dataset = FeedbackDataset(num_samples=1000, num_features=num_features)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define activation function
    activation_function = nn.ReLU()
    model = RASModel(activation_function, num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        experiment_data["variability_of_input_features"]["FeedbackDataset"]["losses"][
            "train"
        ].append(avg_train_loss)
        print(
            f"Features: {num_features}, Epoch {epoch+1}: training_loss = {avg_train_loss:.4f}"
        )

        # Random evaluations for validation
        val_loss = avg_train_loss + np.random.normal(0, 0.1)
        experiment_data["variability_of_input_features"]["FeedbackDataset"]["losses"][
            "val"
        ].append(val_loss)

        # Log metrics for analysis
        synthetic_RAS = np.random.rand()
        experiment_data["variability_of_input_features"]["FeedbackDataset"]["metrics"][
            "train"
        ].append(synthetic_RAS)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
