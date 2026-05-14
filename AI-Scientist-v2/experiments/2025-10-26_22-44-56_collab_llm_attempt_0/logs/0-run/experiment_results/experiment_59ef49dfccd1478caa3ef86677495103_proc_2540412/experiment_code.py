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


# Simple model class for the LLM
class SimpleLLM(nn.Module):
    def __init__(self):
        super(SimpleLLM, self).__init__()
        self.fc = nn.Linear(10, 2)  # Simple linear layer

    def forward(self, x):
        return self.fc(x)


# Synthetic dataset for multi-turn interactions with distinct styles
class SyntheticDataset(Dataset):
    def __init__(self, prompts, responses):
        self.data = list(zip(prompts, responses))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Define different styles of prompts and responses
technical_prompts = [
    "Explain the algorithm of quicksort.",
    "What is a differential equation?",
    "How does a neural network learn?",
]
technical_responses = [
    "Quicksort is a divide-and-conquer algorithm...",
    "A differential equation is an equation involving derivatives...",
    "A neural network learns through adjustments of weights...",
]

casual_prompts = [
    "Hey, what's your favorite movie?",
    "Do you think aliens exist?",
    "What's your idea of a perfect weekend?",
]
casual_responses = [
    "I love action movies!",
    "Of course, the universe is too big for just us!",
    "Chilling at the beach sounds perfect!",
]

formal_prompts = [
    "Could you elucidate the concept of blockchain?",
    "What are the implications of artificial intelligence?",
    "Please discuss the principles of classical mechanics.",
]
formal_responses = [
    "Blockchain is a distributed ledger technology...",
    "The implications of AI include job displacement, ethical considerations...",
    "Classical mechanics is governed by Newton's laws...",
]

# Training parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001
momentum_values = [0.0, 0.5, 0.9]  # Momentum hyperparameter values to tune

# Experiment data storage
experiment_data = {"multi_dataset_variation": {}}


# Function for training a model on a given dataset
def train_model(dataset_name, prompts, responses):
    dataset = SyntheticDataset(prompts, responses)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    experiment_data["multi_dataset_variation"][dataset_name] = {
        "metrics": {"train": [], "CIS": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    for momentum in momentum_values:
        print(f"Training with momentum: {momentum} on dataset: {dataset_name}")

        model = SimpleLLM().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_CIS = 0

            for prompts_batch, responses_batch in dataloader:
                inputs = torch.randn(batch_size, 10).to(device)
                outputs = model(inputs)

                # Generate dummy target responses for loss calculation
                targets = torch.randn(batch_size, 2).to(device)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Dummy CIS calculation (to be replaced by the actual metric)
                CIS = np.random.rand()  # Placeholder for the actual CIS computation
                total_CIS += CIS

            avg_loss = total_loss / len(dataloader)
            avg_CIS = total_CIS / len(dataloader)
            experiment_data["multi_dataset_variation"][dataset_name]["losses"][
                "train"
            ].append(avg_loss)
            experiment_data["multi_dataset_variation"][dataset_name]["metrics"][
                "train"
            ].append(avg_loss)
            experiment_data["multi_dataset_variation"][dataset_name]["metrics"][
                "CIS"
            ].append(avg_CIS)

            print(
                f"Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}, CIS = {avg_CIS:.4f}"
            )


# Train models on all datasets
train_model("technical", technical_prompts, technical_responses)
train_model("casual", casual_prompts, casual_responses)
train_model("formal", formal_prompts, formal_responses)

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
