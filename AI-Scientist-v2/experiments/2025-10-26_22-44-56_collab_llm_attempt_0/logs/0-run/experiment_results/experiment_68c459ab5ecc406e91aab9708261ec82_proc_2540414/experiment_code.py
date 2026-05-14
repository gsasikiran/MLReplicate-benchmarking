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


# Simple LLM model with configurable activation function
class SimpleLLM(nn.Module):
    def __init__(self, activation_fn=nn.ReLU):
        super(SimpleLLM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            activation_fn(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.fc(x)


# Parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001
momentum_values = [0.0, 0.5, 0.9]  # Momentum hyperparameter values to tune
activations = [nn.ReLU, nn.LeakyReLU, nn.Tanh]  # Activation functions to analyze

# Initialize dataset and dataloader
dataset = SyntheticDataset(size=100)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Experiment data storage
experiment_data = {
    "activation_function_analysis": {
        "synthetic_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Training loop for ablation study of activation functions
for activation_fn in activations:
    print(f"Training with activation function: {activation_fn.__name__}")

    model = SimpleLLM(activation_fn).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_cis = 0  # Initialize Collaborative Interaction Score

        for prompts, responses in dataloader:
            inputs = torch.randn(batch_size, 10).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, torch.randn(batch_size, 2).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Simulated CIS metric (this should be replaced with real user engagement data)
            cis = np.random.rand()
            total_cis += cis

        avg_loss = total_loss / len(dataloader)
        avg_cis = total_cis / len(dataloader)  # Calculate average CIS

        experiment_data["activation_function_analysis"]["synthetic_dataset"]["losses"][
            "train"
        ].append(avg_loss)
        experiment_data["activation_function_analysis"]["synthetic_dataset"]["metrics"][
            "train"
        ].append(avg_cis)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}, CIS = {avg_cis:.4f}"
        )

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
