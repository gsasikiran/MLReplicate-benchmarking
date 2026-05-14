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


# Synthetic datasets
class SyntheticDataset(Dataset):
    def __init__(self, size, complexity):
        self.data = []
        for _ in range(size):
            prompt = f"User prompt for task with complexity {complexity}."
            response = f"Model's generated response for complexity {complexity}."
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
momentum_values = [0.0, 0.5, 0.9]
datasets_info = [(100, "simple"), (100, "moderate"), (100, "complex")]

# Experiment data storage
experiment_data = {"multi_dataset_robustness": {}}

# Training loop for different datasets
for dataset_size, complexity in datasets_info:
    experiment_data["multi_dataset_robustness"][complexity] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    print(f"Training on {complexity} dataset")
    dataset = SyntheticDataset(size=dataset_size, complexity=complexity)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for momentum in momentum_values:
        print(f"Training with momentum: {momentum}")

        model = SimpleLLM().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_cis = 0  # Track CIS

            for prompts, responses in dataloader:
                inputs = torch.randn(batch_size, 10).to(device)
                outputs = model(inputs).to(device)

                targets = torch.randn(batch_size, 2).to(device)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Simulated Collaborative Interaction Score (CIS)
                # For simplicity, we simulate it as a random score here.
                cis = np.random.rand()
                total_cis += cis

            avg_loss = total_loss / len(dataloader)
            avg_cis = total_cis / len(dataloader)

            experiment_data["multi_dataset_robustness"][complexity]["losses"][
                "train"
            ].append(avg_loss)
            experiment_data["multi_dataset_robustness"][complexity]["metrics"][
                "train"
            ].append(avg_cis)

            print(
                f"Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}, CIS = {avg_cis:.4f}"
            )

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
