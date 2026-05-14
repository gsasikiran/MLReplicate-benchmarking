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


# Synthetic datasets for different task types
class SentimentAnalysisDataset(Dataset):
    def __init__(self, size):
        self.data = [("I love this product!", 1) for _ in range(size)] + [
            ("I hate this product!", 0) for _ in range(size)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class QuestionAnsweringDataset(Dataset):
    def __init__(self, size):
        self.data = [("What is AI?", "Artificial Intelligence") for _ in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TextSummarizationDataset(Dataset):
    def __init__(self, size):
        self.data = [
            ("The cat sat on the mat.", "A cat on a mat.") for _ in range(size)
        ]

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

# Experiment data storage
experiment_data = {
    "multi_dataset_study": {
        "sentiment_analysis": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "question_answering": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "text_summarization": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}


# Function to train across different datasets
def train_on_dataset(dataset_name, dataset, optimizer, criterion):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

        # Simulated UES metric
        ues = np.random.rand()
        experiment_data["multi_dataset_study"][dataset_name]["losses"]["train"].append(
            total_loss / len(dataloader)
        )
        experiment_data["multi_dataset_study"][dataset_name]["metrics"]["train"].append(
            ues
        )


# Training loop with momentum tuning for each dataset
for momentum in momentum_values:
    print(f"Training with momentum: {momentum}")

    model = SimpleLLM().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    # Train on each dataset
    train_on_dataset(
        "sentiment_analysis", SentimentAnalysisDataset(size=100), optimizer, criterion
    )
    train_on_dataset(
        "question_answering", QuestionAnsweringDataset(size=100), optimizer, criterion
    )
    train_on_dataset(
        "text_summarization", TextSummarizationDataset(size=100), optimizer, criterion
    )

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
