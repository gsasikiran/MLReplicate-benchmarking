import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Model definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# Synthetic data generation with outliers
def generate_data(num_samples, num_features, outlier_fraction=0.0):
    np.random.seed(42)
    # Normal data
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome

    # Introduce outliers
    num_outliers = int(num_samples * outlier_fraction)
    if num_outliers > 0:
        outlier_X = np.random.uniform(
            low=10, high=20, size=(num_outliers, num_features)
        ).astype(np.float32)
        outlier_y = np.random.randint(0, 2, num_outliers).astype(np.float32)
        X = np.vstack((X, outlier_X))
        y = np.concatenate((y, outlier_y))

    return X, y


# Hyperparameter tuning setup
epoch_list = [5, 10, 20, 30]  # Different values for num_epochs
outlier_fractions = [0.0, 0.05, 0.10]
experiment_data = {"Outlier Impact Assessment": {}}

num_samples = 1000
num_features = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for outlier_fraction in outlier_fractions:
    key = f"{int(outlier_fraction * 100)}% Outliers"
    experiment_data["Outlier Impact Assessment"][key] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    X, y = generate_data(num_samples, num_features, outlier_fraction)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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
            experiment_data["Outlier Impact Assessment"][key]["losses"]["train"].append(
                train_loss
            )
            experiment_data["Outlier Impact Assessment"][key]["metrics"][
                "train"
            ].append(train_accuracy)

            # Calculate PAR
            screening_capacity = total_samples
            par = train_accuracy * screening_capacity
            print(
                f"Outlier Fraction: {int(outlier_fraction * 100)}%, Epoch {epoch + 1} of {num_epochs}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}, PAR = {par:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
