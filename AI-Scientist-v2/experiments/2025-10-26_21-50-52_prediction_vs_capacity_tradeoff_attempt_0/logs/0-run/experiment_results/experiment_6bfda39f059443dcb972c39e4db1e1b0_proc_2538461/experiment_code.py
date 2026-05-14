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
X = np.random.randn(num_samples, num_features).astype(np.float32)
y = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome

# Create Dataset and DataLoader
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model definition with weight initialization
class SimpleNN(nn.Module):
    def __init__(self, init_type="xavier"):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 1)
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        if init_type == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif init_type == "he":
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        elif init_type == "normal":
            nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter tuning for weight initialization methods
init_methods = ["xavier", "he", "normal"]
experiment_data = {}

for init_method in init_methods:
    model = SimpleNN(init_type=init_method).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    experiment_data[init_method] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    num_epochs = 10
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
        experiment_data[init_method]["losses"]["train"].append(train_loss)
        experiment_data[init_method]["metrics"]["train"].append(train_accuracy)

        # Calculate PAR
        screening_capacity = total_samples
        par = train_accuracy * screening_capacity
        print(
            f"[{init_method}] Epoch {epoch + 1}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}, PAR = {par:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
