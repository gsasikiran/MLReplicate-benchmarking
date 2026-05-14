import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Function to create synthetic datasets
def create_dataset(seed, sample_size):
    np.random.seed(seed)
    X = np.random.rand(sample_size, 3)
    y = (
        0.5 * X[:, 0]
        + 0.3 * X[:, 1]
        + 0.2 * X[:, 2]
        + np.random.normal(0, 0.1, sample_size)
    )
    return X, y


# Generate multiple datasets
datasets = {
    "dataset_1": create_dataset(seed=0, sample_size=1000),
    "dataset_2": create_dataset(seed=1, sample_size=1000),
    "dataset_3": create_dataset(seed=2, sample_size=1000),
}

# Normalize and prepare data
experiment_data = {"multi_dataset_evaluation": {}}
for dataset_name, (X, y) in datasets.items():
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Define the model
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(3, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    batch_sizes = [16, 32, 64]
    experiment_data["multi_dataset_evaluation"][dataset_name] = {}

    for batch_size in batch_sizes:
        experiment_data["multi_dataset_evaluation"][dataset_name][batch_size] = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        model = SimpleNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )

        epochs = 100
        for epoch in range(epochs):
            # Train
            model.train()
            for data in train_loader:
                inputs, targets = data
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                train_loss = criterion(outputs, targets)
                train_loss.backward()
                optimizer.step()

            experiment_data["multi_dataset_evaluation"][dataset_name][batch_size][
                "losses"
            ]["train"].append(train_loss.item())

            # Validate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor)
                experiment_data["multi_dataset_evaluation"][dataset_name][batch_size][
                    "losses"
                ]["val"].append(val_loss.item())

            PWIS = 1 - val_loss.item()  # Calculate PWIS
            experiment_data["multi_dataset_evaluation"][dataset_name][batch_size][
                "metrics"
            ]["val"].append(PWIS)
            experiment_data["multi_dataset_evaluation"][dataset_name][batch_size][
                "predictions"
            ].append(val_outputs.cpu().numpy())
            experiment_data["multi_dataset_evaluation"][dataset_name][batch_size][
                "ground_truth"
            ].append(y_val_tensor.cpu().numpy())

            print(
                f"{dataset_name}, Batch Size {batch_size}, Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss.item():.4f}, Validation Loss = {val_loss.item():.4f}, PWIS = {PWIS:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
