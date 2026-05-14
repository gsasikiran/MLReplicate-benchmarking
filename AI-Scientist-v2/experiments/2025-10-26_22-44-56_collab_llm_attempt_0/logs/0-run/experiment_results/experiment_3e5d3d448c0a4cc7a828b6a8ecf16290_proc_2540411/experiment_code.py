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


# Synthetic dataset for varying input sizes
class SyntheticDataset(Dataset):
    def __init__(self, size, input_size):
        self.input_size = input_size
        self.data = []
        for _ in range(size):
            prompt = "User prompt for task."
            response = "Model's generated response."
            self.data.append((prompt, response))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Generating random input based on the input size
        input_data = torch.randn(self.input_size)
        return input_data, torch.randn(2)  # Simulated output


# Simple LLM model
class SimpleLLM(nn.Module):
    def __init__(self, input_size):
        super(SimpleLLM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
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
input_sizes = [5, 10, 15]  # Different input resolutions

# Experiment data storage
experiment_data = {
    "multi_resolution_input_analysis": {},
}

# Iterate over different input sizes
for input_size in input_sizes:
    experiment_data["multi_resolution_input_analysis"][f"input_size_{input_size}"] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Initialize dataset and dataloader for the current input size
    dataset = SyntheticDataset(size=100, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop with momentum tuning
    for momentum in momentum_values:
        print(f"Training with momentum: {momentum} on input size: {input_size}")

        model = SimpleLLM(input_size=input_size).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            experiment_data["multi_resolution_input_analysis"][
                f"input_size_{input_size}"
            ]["losses"]["train"].append(avg_loss)

            # Simulated UES metric (this should be replaced with a real calculation in practice)
            ues = np.random.rand()
            experiment_data["multi_resolution_input_analysis"][
                f"input_size_{input_size}"
            ]["metrics"]["train"].append(ues)

            print(
                f"Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}, UES = {ues:.4f}"
            )

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
