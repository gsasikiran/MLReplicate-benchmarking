import os
import subprocess
import sys

# Check if sklearn is installed
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import PolynomialFeatures
except ModuleNotFoundError:
    print("sklearn not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import PolynomialFeatures

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic data generation
np.random.seed(42)
num_samples = 1000
num_features = 10
X = np.random.randn(num_samples, num_features).astype(np.float32)
y = (np.random.random(num_samples) > 0.5).astype(np.float32)  # Binary outcome

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Create Dataset and DataLoader for original dataset
dataset_original = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
data_loader_original = DataLoader(dataset_original, batch_size=32, shuffle=True)

# Create Dataset and DataLoader for polynomial dataset
dataset_poly = TensorDataset(torch.from_numpy(X_poly), torch.from_numpy(y))
data_loader_poly = DataLoader(dataset_poly, batch_size=32, shuffle=True)


# Model definition
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter tuning setup
epoch_list = [5, 10, 20, 30]
experiment_data = {
    "ablation_study_feature_engineering": {
        "original_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
            "par_values": [],
        },
        "polynomial_dataset": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
            "par_values": [],
        },
    }
}

for num_epochs in epoch_list:
    # Training on original dataset
    model_original = SimpleNN(num_features).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_original.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model_original.train()
        epoch_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in data_loader_original:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_original(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_correct += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)

        train_loss = epoch_loss / len(data_loader_original)
        train_accuracy = total_correct / total_samples
        par = total_correct / (total_samples + 1e-6)  # Avoid division by zero
        experiment_data["ablation_study_feature_engineering"]["original_dataset"][
            "losses"
        ]["train"].append(train_loss)
        experiment_data["ablation_study_feature_engineering"]["original_dataset"][
            "metrics"
        ]["train"].append(train_accuracy)
        experiment_data["ablation_study_feature_engineering"]["original_dataset"][
            "par_values"
        ].append(par)

        print(
            f"Epoch {epoch + 1}: original train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}, PAR = {par:.4f}"
        )

    # Training on polynomial dataset
    model_poly = SimpleNN(X_poly.shape[1]).to(device)
    optimizer = optim.Adam(model_poly.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model_poly.train()
        epoch_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in data_loader_poly:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model_poly(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_correct += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)

        train_loss = epoch_loss / len(data_loader_poly)
        train_accuracy = total_correct / total_samples
        par = total_correct / (total_samples + 1e-6)
        experiment_data["ablation_study_feature_engineering"]["polynomial_dataset"][
            "losses"
        ]["train"].append(train_loss)
        experiment_data["ablation_study_feature_engineering"]["polynomial_dataset"][
            "metrics"
        ]["train"].append(train_accuracy)
        experiment_data["ablation_study_feature_engineering"]["polynomial_dataset"][
            "par_values"
        ].append(par)

        print(
            f"Epoch {epoch + 1}: polynomial train_loss = {train_loss:.4f}, train_accuracy = {train_accuracy:.4f}, PAR = {par:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
