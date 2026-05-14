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
    def __init__(self):
        super(SimpleLLM, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),  # Simple input layer
            nn.ReLU(),
            nn.Linear(64, 2),  # Output layer for response prediction
        )

    def forward(self, x):
        return self.fc(x)


# Parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# Initialize dataset and dataloader
dataset = SyntheticDataset(size=100)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer and loss function
model = SimpleLLM().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Experiment data storage
experiment_data = {
    "multi_turn_interactions": {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for prompts, responses in dataloader:
        # Dummy data conversion (normally use tokenized inputs)
        inputs = torch.randn(batch_size, 10).to(device)

        # Compute predictions
        outputs = model(inputs)

        # Calculate loss (dummy response placeholders for demonstration)
        loss = criterion(outputs, torch.randn(batch_size, 2).to(device))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    experiment_data["multi_turn_interactions"]["losses"]["train"].append(avg_loss)

    # Simulated UES metric (this should be replaced with a real calculation in practice)
    ues = np.random.rand()
    experiment_data["multi_turn_interactions"]["metrics"]["train"].append(ues)

    print(f"Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}, UES = {ues:.4f}")

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
