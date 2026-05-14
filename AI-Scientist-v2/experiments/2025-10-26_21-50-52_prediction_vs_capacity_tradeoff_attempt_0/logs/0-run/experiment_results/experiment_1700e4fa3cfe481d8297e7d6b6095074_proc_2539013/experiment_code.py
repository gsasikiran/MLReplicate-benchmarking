import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Define input feature dimensions
feature_dims = [5, 10, 15]
num_samples = 1000
epoch_list = [5, 10, 20, 30]  # Different values for num_epochs

# Experiment data structure
experiment_data = {"input_feature_variation": {}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Model definition
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


for num_features in feature_dims:
    # Synthetic data generation
    np.random.seed(42)
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    experiment_data["input_feature_variation"][f"{num_features}_features"] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    for num_epochs in epoch_list:
        model = SimpleNN(num_features).to(
            device
        )  # Initialize model for current feature set
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
            experiment_data["input_feature_variation"][f"{num_features}_features"][
                "losses"
            ]["train"].append(train_loss)
            experiment_data["input_feature_variation"][f"{num_features}_features"][
                "metrics"
            ]["train"].append(train_accuracy)

            print(
                f"[Features: {num_features}] Epoch {epoch + 1} of {num_epochs}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}"
            )

    # Save predictions and ground truth for later analysis
    # Here we would normally test after training and store those results
    experiment_data["input_feature_variation"][f"{num_features}_features"][
        "predictions"
    ] = []
    experiment_data["input_feature_variation"][f"{num_features}_features"][
        "ground_truth"
    ] = y.tolist()

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
