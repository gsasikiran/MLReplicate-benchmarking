import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Function to create synthetic data
def create_synthetic_data(n_samples, n_features, n_classes, random_state):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_classes=n_classes,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


# Create three different synthetic datasets
datasets = {
    "dataset_1": create_synthetic_data(1000, 10, 2, 42),
    "dataset_2": create_synthetic_data(500, 20, 2, 43),
    "dataset_3": create_synthetic_data(
        2000, 15, 2, 44
    ),  # update classes to 2 for binary classification
}


# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 1)  # single output node for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# Initialize experiment data storage
experiment_data = {"multiple_synthetic_datasets": {}}

# List of learning rates to test
learning_rates = [0.001, 0.01, 0.1]

# Training on each synthetic dataset
for dataset_name, (X_train, X_val, y_train, y_val) in datasets.items():
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = (
        torch.FloatTensor(y_train).view(-1, 1).to(device)
    )  # Ensuring proper target shape
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = (
        torch.FloatTensor(y_val).view(-1, 1).to(device)
    )  # Ensuring proper target shape

    experiment_data["multiple_synthetic_datasets"][dataset_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "learning_rates": [],
    }

    for lr in learning_rates:
        print(f"Training {dataset_name} with learning rate: {lr}")
        model = SimpleNN(X_train.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor).view(
                -1, 1
            )  # Reshape outputs to match target shape
            train_loss = criterion(outputs, y_train_tensor)

            train_loss.backward()
            optimizer.step()
            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "train"
            ].append(train_loss.item())

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).view(
                    -1, 1
                )  # Reshape outputs to match target shape
                val_loss = criterion(val_outputs, y_val_tensor)

            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "val"
            ].append(val_loss.item())
            print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

            # Metrics calculation
            train_preds = (outputs > 0.5).float()
            val_preds = (val_outputs > 0.5).float()

            train_score = (train_preds == y_train_tensor).float().mean().item()
            val_score = (val_preds == y_val_tensor).float().mean().item()

            experiment_data["multiple_synthetic_datasets"][dataset_name]["metrics"][
                "train"
            ].append(train_score)
            experiment_data["multiple_synthetic_datasets"][dataset_name]["metrics"][
                "val"
            ].append(val_score)

        experiment_data["multiple_synthetic_datasets"][dataset_name][
            "learning_rates"
        ].append(lr)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
