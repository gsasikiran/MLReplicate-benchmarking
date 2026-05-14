# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity

# Constants
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
num_samples = 1000
seq_length = 10
num_classes = 5
embedding_dim = 16
hidden_dim = 32
num_epochs = 50
batch_size = 32
learning_rate = 0.001
dropout_rates = [0.0, 0.2, 0.4, 0.6, 0.8]  # Range of dropout rates
noise_levels = {"no_noise": 0, "low_noise": 1, "moderate_noise": 3, "high_noise": 5}


# Create synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.data = torch.randint(0, num_classes, (num_samples, seq_length))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.data[index],
            self.data[index],
        )  # Returning input and target as the same for simplicity


# Model Definition with Dropout
class LSTMModel(nn.Module):
    def __init__(self, dropout_rate):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, dropout=dropout_rate
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out)


# Metrics Storage
experiment_data = {
    "input_noise_variation": {
        "synthetic_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# Evaluation Metric: CODS
def calculate_cods(outputs):
    similarity_matrix = cosine_similarity(outputs)
    distinctiveness = 1 - np.mean(similarity_matrix)
    return distinctiveness


# Training Routine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = SyntheticDataset(num_samples, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for noise_name, noise_value in noise_levels.items():
    for dropout_rate in dropout_rates:
        model = LSTMModel(dropout_rate=dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print(f"Training with noise: {noise_name} and dropout rate: {dropout_rate}")

        for epoch in range(num_epochs):
            epoch_loss = 0
            all_outputs = []

            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Add systematic noise
                noise = (
                    torch.randint(0, num_classes, inputs.shape).to(device) * noise_value
                )
                inputs = (inputs + noise) % num_classes  # Ensure valid index ranges

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(
                    outputs.view(-1, num_classes), targets.view(-1)
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                all_outputs.append(outputs.detach().cpu().numpy())

            avg_loss = epoch_loss / len(dataloader)
            experiment_data["input_noise_variation"]["synthetic_dataset"]["losses"][
                "train"
            ].append(avg_loss)

            # Calculate CODS
            all_outputs = np.concatenate(all_outputs)
            cods = calculate_cods(all_outputs.reshape(-1, num_classes))
            experiment_data["input_noise_variation"]["synthetic_dataset"]["metrics"][
                "train"
            ].append(cods)

            print(f"Epoch {epoch+1}: training_loss = {avg_loss:.4f}, CODS = {cods:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
