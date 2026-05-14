import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic data generation
np.random.seed(42)
num_samples = 1000
num_features = 10

# Dataset 1: Normal distribution
X1 = np.random.randn(num_samples, num_features).astype(np.float32)
y1 = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome

# Dataset 2: Uniform distribution
X2 = np.random.uniform(low=-1, high=1, size=(num_samples, num_features)).astype(
    np.float32
)
y2 = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome

# Dataset 3: Skewed distribution
X3 = np.random.exponential(scale=1.0, size=(num_samples, num_features)).astype(
    np.float32
)
y3 = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome

# Creating Datasets and DataLoaders
datasets = {
    "normal_dataset": TensorDataset(torch.from_numpy(X1), torch.from_numpy(y1)),
    "uniform_dataset": TensorDataset(torch.from_numpy(X2), torch.from_numpy(y2)),
    "skewed_dataset": TensorDataset(torch.from_numpy(X3), torch.from_numpy(y3)),
}

data_loaders = {
    name: DataLoader(dataset, batch_size=32, shuffle=True)
    for name, dataset in datasets.items()
}


# Model definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter tuning setup
epoch_list = [5, 10, 20, 30]  # Different values for num_epochs
experiment_data = {
    "multiple_synthetic_datasets_variation": {
        "normal_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "uniform_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "skewed_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

for dataset_name, data_loader in data_loaders.items():
    for num_epochs in epoch_list:
        model = SimpleNN().to(
            device
        )  # Reinitialize the model for each dataset and each epoch
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_correct += (predicted == batch_y).sum().item()
                total_samples += batch_y.size(0)

            train_loss = epoch_loss / len(data_loader)
            train_accuracy = total_correct / total_samples
            experiment_data["multiple_synthetic_datasets_variation"][dataset_name][
                "losses"
            ]["train"].append(train_loss)
            experiment_data["multiple_synthetic_datasets_variation"][dataset_name][
                "metrics"
            ]["train"].append(train_accuracy)

            # Store predictions and ground truth
            experiment_data["multiple_synthetic_datasets_variation"][dataset_name][
                "predictions"
            ].append(predicted.cpu().numpy())
            experiment_data["multiple_synthetic_datasets_variation"][dataset_name][
                "ground_truth"
            ].append(batch_y.cpu().numpy())

            # Print results
            par = train_accuracy * total_samples
            print(
                f"{dataset_name.capitalize()} - Epoch {epoch + 1}/{num_epochs}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}, PAR = {par:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
