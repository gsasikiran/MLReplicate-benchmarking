import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset

# Setting up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleNN(nn.Module):
    def __init__(self, activation_fn):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


# Generate synthetic dataset
class PeerReviewDataset(Dataset):
    def __init__(self, size):
        self.data = np.random.rand(size, 2)
        self.labels = np.random.rand(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.data[idx], dtype=torch.float32).to(device),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32).to(device),
        }


# Load datasets from Hugging Face (not used for the ablation study, just included for completeness)
dataset_1 = load_dataset("glue", "mrpc", split="train")
dataset_2 = load_dataset("glue", "cola", split="train")

# Create synthetic dataset
dataset = PeerReviewDataset(size=1000)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Hyperparameter tuning for learning rates
learning_rates = [0.001, 0.01, 0.1]
batch_size = 64  # Static batch size for simplicity

# Activation functions to compare
activation_functions = {
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(),
    "ELU": nn.ELU(),
    "Tanh": nn.Tanh(),
}

experiment_data = {
    "activation_function_comparison": {
        "metrics": {},
        "losses": {},
        "predictions": {},
        "ground_truth": {},
    }
}

epochs = 10

for activation_name, activation_fn in activation_functions.items():
    # Initialize data storage for this activation function
    experiment_data["activation_function_comparison"]["metrics"][activation_name] = {
        "train": [],
        "val": [],
    }
    experiment_data["activation_function_comparison"]["losses"][activation_name] = {
        "train": [],
        "val": [],
    }
    experiment_data["activation_function_comparison"]["predictions"][
        activation_name
    ] = []
    experiment_data["activation_function_comparison"]["ground_truth"][
        activation_name
    ] = []

    for lr in learning_rates:
        print(f"Training {activation_name} with learning rate: {lr}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleNN(activation_fn).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                features, labels = batch["features"], batch["label"]
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            experiment_data["activation_function_comparison"]["losses"][
                activation_name
            ]["train"].append(avg_train_loss)

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    features, labels = batch["features"], batch["label"]
                    outputs = model(features).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            experiment_data["activation_function_comparison"]["losses"][
                activation_name
            ]["val"].append(avg_val_loss)
            rqi = 1 - avg_val_loss
            experiment_data["activation_function_comparison"]["metrics"][
                activation_name
            ]["train"].append(rqi)

            print(
                f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}, RQI = {rqi:.4f}"
            )

        # Collect predictions and ground truths
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                features, labels = batch["features"], batch["label"]
                outputs = model(features).squeeze()
                experiment_data["activation_function_comparison"]["predictions"][
                    activation_name
                ].extend(outputs.cpu().numpy())
                experiment_data["activation_function_comparison"]["ground_truth"][
                    activation_name
                ].extend(labels.cpu().numpy())

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
