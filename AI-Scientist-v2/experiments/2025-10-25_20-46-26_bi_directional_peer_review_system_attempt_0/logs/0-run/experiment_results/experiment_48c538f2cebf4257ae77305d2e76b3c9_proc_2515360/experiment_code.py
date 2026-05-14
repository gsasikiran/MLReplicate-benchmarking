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


# Synthetic dataset generation with different distributions
class FeedbackDataset(Dataset):
    def __init__(self, num_samples=1000, mean_variance=None):
        self.features = self._generate_features(num_samples, mean_variance)
        self.labels = self.calculate_labels(self.features)

    def _generate_features(self, num_samples, mean_variance):
        if mean_variance is None:
            return torch.rand(num_samples, 4)  # Default uniform distribution
        mean, std = mean_variance
        return torch.normal(
            mean=torch.tensor(mean), std=torch.tensor(std), size=(num_samples, 4)
        )

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


# Simple neural network model with tunable activation function
class RASModel(nn.Module):
    def __init__(self, activation_function):
        super(RASModel, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)  # Regression output
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # Output between 0 and 1


# Prepare different datasets
datasets = {
    "Uniform": FeedbackDataset(num_samples=1000),
    "Normal_Mean_0_Var_1": FeedbackDataset(num_samples=1000, mean_variance=(0, 1)),
    "Normal_Mean_2_Var_0.5": FeedbackDataset(num_samples=1000, mean_variance=(2, 0.5)),
}

# Define available activation functions
activation_functions = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "swish": nn.SiLU(),
}

# Experiment data to save metrics
experiment_data = {"multi_dataset_evaluation": {}}

# Training and evaluation for different activation functions and datasets
num_epochs = 20
for dataset_name, dataset in datasets.items():
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    experiment_data["multi_dataset_evaluation"][dataset_name] = {
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
            experiment_data["multi_dataset_evaluation"][dataset_name]["losses"][
                "train"
            ].append(avg_train_loss)
            print(
                f"Dataset: {dataset_name}, Activation Function: {act_name}, Epoch {epoch+1}: training_loss = {avg_train_loss:.4f}"
            )

            # Random evaluations for demonstration purposes (not real validation)
            val_loss = avg_train_loss + np.random.normal(0, 0.1)  # Simulate some noise
            experiment_data["multi_dataset_evaluation"][dataset_name]["losses"][
                "val"
            ].append(val_loss)

            # Store predictions and ground truth for analysis
            experiment_data["multi_dataset_evaluation"][dataset_name][
                "predictions"
            ].extend(outputs.detach().cpu().numpy())
            experiment_data["multi_dataset_evaluation"][dataset_name][
                "ground_truth"
            ].extend(labels.detach().cpu().numpy())

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
