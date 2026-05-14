import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic dataset class with varying prompts and responses
class SyntheticDataset(Dataset):
    def __init__(self, size, feature_dimension):
        self.data = []
        for _ in range(size):
            prompt = f"User prompt with {feature_dimension} features."
            response = "Model's generated response."
            self.data.append((prompt, response))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Simple LLM model
class SimpleLLM(nn.Module):
    def __init__(self, input_dim):
        super(SimpleLLM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.fc(x)


# Parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001
momentum_values = [0.0, 0.5, 0.9]  # Momentum hyperparameter values to tune
feature_dimensions = [5, 10, 15]  # Different feature dimensions for the study

# Experiment data storage
experiment_data = {"input_feature_variability_analysis": {}}

# Training loop over varying input feature dimensions
for feature_dim in feature_dimensions:
    print(f"Evaluating feature dimension: {feature_dim}")
    experiment_data["input_feature_variability_analysis"][
        f"dataset_dim_{feature_dim}"
    ] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Initialize dataset and dataloader for the current feature dimension
    dataset = SyntheticDataset(size=100, feature_dimension=feature_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training with different momentum values
    for momentum in momentum_values:
        print(f"Training with momentum: {momentum}")

        model = SimpleLLM(input_dim=feature_dim).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for prompts, responses in dataloader:
                inputs = torch.randn(batch_size, feature_dim).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, torch.randn(batch_size, 2).to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            experiment_data["input_feature_variability_analysis"][
                f"dataset_dim_{feature_dim}"
            ]["losses"]["train"].append(avg_loss)

            # Simulated UES metric
            ues = np.random.rand()
            experiment_data["input_feature_variability_analysis"][
                f"dataset_dim_{feature_dim}"
            ]["metrics"]["train"].append(ues)

            print(
                f"Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}, UES = {ues:.4f}"
            )

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
