import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic data generation
np.random.seed(42)
num_samples = 1000
num_features = 10
X = np.random.randn(num_samples, num_features).astype(np.float32)
y = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome

# Create Dataset and DataLoader
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model definition with different output layer variations
class SimpleNNSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        super(SimpleNNSoftmax, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 2)  # 2 outputs for binary classification (one-hot)
        self.temperature = temperature  # Save temperature as an instance variable

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return torch.softmax(logits / self.temperature, dim=1)


temperature_list = [1.0, 0.5, 2.0]
experiment_data = {
    "output_layer_variation": {
        "synthetic_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

for temperature in temperature_list:
    model = SimpleNNSoftmax(temperature).to(
        device
    )  # Reinitialize the model for each temperature
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):  # Fixed number of epochs for comparison
        model.train()
        epoch_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in data_loader:
            batch_X, batch_y = (
                batch_X.to(device),
                batch_y.to(device).long(),
            )  # Convert to long for CrossEntropy
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)

        train_loss = epoch_loss / len(data_loader)
        train_accuracy = total_correct / total_samples
        experiment_data["output_layer_variation"]["synthetic_dataset"]["losses"][
            "train"
        ].append(train_loss)
        experiment_data["output_layer_variation"]["synthetic_dataset"]["metrics"][
            "train"
        ].append(train_accuracy)

        # Collect predictions and ground truth
        experiment_data["output_layer_variation"]["synthetic_dataset"][
            "predictions"
        ].extend(predicted.cpu().numpy())
        experiment_data["output_layer_variation"]["synthetic_dataset"][
            "ground_truth"
        ].extend(batch_y.cpu().numpy().tolist())

        # Calculate PAR
        screening_capacity = total_samples
        par = train_accuracy * screening_capacity
        print(
            f"Temperature {temperature}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}, PAR = {par:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
