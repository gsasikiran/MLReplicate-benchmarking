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
num_epochs_list = [50, 100, 150]  # Hyperparameter to tune
batch_size = 32
learning_rate = 0.001


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


# Model Definition
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out)


# Metrics Storage
experiment_data = {
    "num_epochs_tuning": {
        "synthetic_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
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

for num_epochs in num_epochs_list:
    model = LSTMModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0
        all_outputs = []

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Seed-conditioning: introducing randomness with constraint
            noise = torch.randint(0, num_classes, inputs.shape).to(device)
            inputs = (inputs + noise) % num_classes  # Ensure valid index ranges

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_outputs.append(outputs.detach().cpu().numpy())

        avg_loss = epoch_loss / len(dataloader)
        experiment_data["num_epochs_tuning"]["synthetic_dataset"]["losses"][
            "train"
        ].append(avg_loss)

        # Calculate CODS
        all_outputs = np.concatenate(all_outputs)
        cods = calculate_cods(all_outputs.reshape(-1, num_classes))
        experiment_data["num_epochs_tuning"]["synthetic_dataset"]["metrics"][
            "train"
        ].append(cods)

        print(
            f"Epoch {epoch+1}/{num_epochs}: training_loss = {avg_loss:.4f}, CODS = {cods:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
