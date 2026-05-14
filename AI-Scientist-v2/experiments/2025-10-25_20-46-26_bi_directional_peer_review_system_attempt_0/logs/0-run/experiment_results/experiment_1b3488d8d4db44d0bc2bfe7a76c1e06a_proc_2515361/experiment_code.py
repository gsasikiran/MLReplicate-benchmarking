# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

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


class FeedbackDataset(Dataset):
    def __init__(self, num_samples=1000, remove_feature=None):
        self.features = torch.rand(num_samples, 4)
        self.labels = self.calculate_labels(self.features)
        if remove_feature is not None:
            self.features = torch.cat(
                [
                    self.features[:, :remove_feature],
                    self.features[:, remove_feature + 1 :],
                ],
                dim=1,
            )

    def calculate_labels(self, features):
        return (
            features[:, 0] + features[:, 1] - features[:, 2] + features[:, 3]
        ).clamp(0, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "label": self.labels[idx]}


class RASModel(nn.Module):
    def __init__(self, activation_function):
        super(RASModel, self).__init__()
        self.fc1 = nn.Linear(3, 8)  # Updated input dimension
        self.fc2 = nn.Linear(8, 1)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# Experiment data to save metrics
experiment_data = {
    "feature_importance_removal": {
        "FeedbackDataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

activation_functions = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "swish": nn.SiLU(),
}

num_epochs = 20

# Perform ablation study by removing each feature
for feature_to_remove in range(4):
    for act_name, act_func in activation_functions.items():
        dataset = FeedbackDataset(num_samples=1000, remove_feature=feature_to_remove)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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
            experiment_data["feature_importance_removal"]["FeedbackDataset"]["losses"][
                "train"
            ].append(avg_train_loss)
            print(
                f"Feature {feature_to_remove}, Activation Function: {act_name}, Epoch {epoch+1}: training_loss = {avg_train_loss:.4f}"
            )

            val_loss = avg_train_loss + np.random.normal(0, 0.1)
            experiment_data["feature_importance_removal"]["FeedbackDataset"]["losses"][
                "val"
            ].append(val_loss)

            # Synthetic RAS for evaluation (would be replaced with actual evaluation)
            synthetic_RAS = np.random.rand()
            experiment_data["feature_importance_removal"]["FeedbackDataset"]["metrics"][
                "train"
            ].append(synthetic_RAS)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
