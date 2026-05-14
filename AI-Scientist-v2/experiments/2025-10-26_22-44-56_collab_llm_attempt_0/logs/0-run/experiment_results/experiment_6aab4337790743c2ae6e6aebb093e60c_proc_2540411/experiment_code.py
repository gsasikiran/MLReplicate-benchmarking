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
    def __init__(self, size, prompt_variation):
        self.data = []
        for _ in range(size):
            prompt = f"User prompt variation: {prompt_variation}."
            response = f"Model's generated response for variation {prompt_variation}."
            self.data.append((prompt, response))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Simple LLM model
class SimpleLLM(nn.Module):
    def __init__(self):
        super(SimpleLLM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
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

# Initialize datasets and dataloaders
datasets = {
    "dataset_1": SyntheticDataset(size=100, prompt_variation=1),
    "dataset_2": SyntheticDataset(size=100, prompt_variation=2),
    "dataset_3": SyntheticDataset(size=100, prompt_variation=3),
}

dataloaders = {
    name: DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for name, dataset in datasets.items()
}

# Experiment data storage
experiment_data = {
    "multi_dataset_evaluation": {
        "dataset_1": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_2": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_3": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Training loop for multiple datasets
for momentum in momentum_values:
    print(f"Training with momentum: {momentum}")

    for dataset_name, dataloader in dataloaders.items():
        model = SimpleLLM().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for prompts, responses in dataloader:
                inputs = torch.randn(batch_size, 10).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, torch.randn(batch_size, 2).to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            experiment_data["multi_dataset_evaluation"][dataset_name]["losses"][
                "train"
            ].append(avg_loss)

            # Simulated UES metric
            ues = np.random.rand()
            experiment_data["multi_dataset_evaluation"][dataset_name]["metrics"][
                "train"
            ].append(ues)

            print(
                f"{dataset_name} - Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}, UES = {ues:.4f}"
            )

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
