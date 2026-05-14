import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Synthetic dataset generation
def generate_datasets(num_samples, num_features):
    datasets = {}

    # Uniform distribution
    X_uniform = np.random.uniform(0, 1, (num_samples, num_features)).astype(np.float32)
    y_uniform = (np.random.random(num_samples) > 0.5).astype(np.float32)
    datasets["uniform"] = TensorDataset(
        torch.from_numpy(X_uniform), torch.from_numpy(y_uniform)
    )

    # Gaussian distribution
    X_gaussian = np.random.randn(num_samples, num_features).astype(np.float32)
    y_gaussian = (np.random.random(num_samples) > 0.5).astype(np.float32)
    datasets["gaussian"] = TensorDataset(
        torch.from_numpy(X_gaussian), torch.from_numpy(y_gaussian)
    )

    # Imbalanced binary outcomes
    X_imbalanced = np.random.randn(num_samples, num_features).astype(np.float32)
    y_imbalanced = np.concatenate(
        [np.zeros(int(0.9 * num_samples)), np.ones(int(0.1 * num_samples))]
    ).astype(np.float32)
    np.random.shuffle(y_imbalanced)  # Shuffle to mix classes
    datasets["imbalanced"] = TensorDataset(
        torch.from_numpy(X_imbalanced), torch.from_numpy(y_imbalanced)
    )

    return datasets


num_samples = 1000
num_features = 10
datasets = generate_datasets(num_samples, num_features)


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
experiment_data = {"multiple_synthetic_datasets": {}}

for dataset_name, dataset in datasets.items():
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    experiment_data["multiple_synthetic_datasets"][dataset_name] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    for num_epochs in epoch_list:
        model = SimpleNN().to(device)  # Reinitialize the model for each epoch
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
            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "train"
            ].append(train_loss)
            experiment_data["multiple_synthetic_datasets"][dataset_name]["metrics"][
                "train"
            ].append(train_accuracy)

            # Save predictions and ground_truth for further analysis
            experiment_data["multiple_synthetic_datasets"][dataset_name][
                "predictions"
            ].extend(predicted.cpu().detach().numpy())
            experiment_data["multiple_synthetic_datasets"][dataset_name][
                "ground_truth"
            ].extend(batch_y.cpu().detach().numpy())

            # Calculate PAR
            screening_capacity = total_samples
            par = train_accuracy * screening_capacity
            print(
                f"Dataset: {dataset_name}, Epoch {epoch + 1} of {num_epochs}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}, PAR = {par:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
