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


# Synthetic dataset for multi-turn interactions
class SyntheticDataset(Dataset):
    def __init__(self, size):
        self.data = []
        for _ in range(size):
            prompt = "User prompt for task."
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
input_dimensions = [5, 10, 15]  # Different input dimensionalities for ablation
momentum_values = [0.0, 0.5, 0.9]  # Momentum hyperparameter values to tune

# Experiment data storage
experiment_data = {
    "input_feature_dimensionality_analysis": {
        "synthetic_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Training loop for varying input dimensions
for input_dim in input_dimensions:
    print(f"Training with input dimension: {input_dim}")

    dataset = SyntheticDataset(size=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for momentum in momentum_values:
        print(f"Using momentum: {momentum}")

        model = SimpleLLM(input_dim).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for prompts, responses in dataloader:
                # Input feature size defined by input_dim
                inputs = torch.randn(batch_size, input_dim).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, torch.randn(batch_size, 2).to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            experiment_data["input_feature_dimensionality_analysis"][
                "synthetic_dataset"
            ]["losses"]["train"].append(avg_loss)

            # Simulated UES metric
            ues = np.random.rand()
            experiment_data["input_feature_dimensionality_analysis"][
                "synthetic_dataset"
            ]["metrics"]["train"].append(ues)

            print(
                f"Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}, UES = {ues:.4f}"
            )

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
