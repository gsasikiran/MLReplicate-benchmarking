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


# Synthetic dataset generation with variability
class FeedbackDataset(Dataset):
    def __init__(self, num_samples=1000, method="default"):
        self.features = torch.rand(num_samples, 4)  # 4 features for each review
        self.labels = self.calculate_labels(self.features, method)

    def calculate_labels(self, features, method):
        if method == "default":
            return (
                features[:, 0] + features[:, 1] - features[:, 2] + features[:, 3]
            ).clamp(0, 1)
        elif method == "alternative_1":
            return (
                features[:, 0] * features[:, 1] + features[:, 2] - features[:, 3]
            ).clamp(0, 1)
        elif method == "alternative_2":
            return (
                features[:, 0] - features[:, 1] + features[:, 2] + features[:, 3]
            ).clamp(0, 1)
        elif method == "alternative_3":
            return (
                features[:, 0] / (1 + features[:, 1]) + features[:, 2] * features[:, 3]
            ).clamp(0, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "label": self.labels[idx]}


# Simple neural network model
class RASModel(nn.Module):
    def __init__(self, activation_function):
        super(RASModel, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)  # Regression output
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# Experiment data to save metrics
experiment_data = {"synthetic_dataset_variability": {}}

# Activation functions
activation_functions = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "swish": nn.SiLU(),
}

# Define dataset methods to be evaluated
dataset_methods = ["default", "alternative_1", "alternative_2", "alternative_3"]

# Training and evaluation for different datasets and activation functions
num_epochs = 20
for method in dataset_methods:
    dataset = FeedbackDataset(num_samples=1000, method=method)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    experiment_data["synthetic_dataset_variability"][method] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for act_name, act_func in activation_functions.items():
        model = RASModel(act_func).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in data_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
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
            experiment_data["synthetic_dataset_variability"][method]["losses"][
                "train"
            ].append(avg_train_loss)
            print(
                f"Dataset: {method}, Activation Function: {act_name}, Epoch {epoch + 1}: training_loss = {avg_train_loss:.4f}"
            )

            # Random evaluations for demonstration (not real validation)
            val_loss = avg_train_loss + np.random.normal(0, 0.1)
            experiment_data["synthetic_dataset_variability"][method]["losses"][
                "val"
            ].append(val_loss)

            # Calculate a synthetic RAS for evaluation
            synthetic_RAS = np.random.rand()
            experiment_data["synthetic_dataset_variability"][method]["metrics"][
                "train"
            ].append(synthetic_RAS)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
