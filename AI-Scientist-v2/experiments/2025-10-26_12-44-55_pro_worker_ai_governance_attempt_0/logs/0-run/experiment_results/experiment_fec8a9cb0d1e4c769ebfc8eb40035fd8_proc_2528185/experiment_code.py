import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic dataset
np.random.seed(42)
data_size = 1000
job_displacement = np.random.uniform(0, 1, data_size)
wage_change = np.random.uniform(-1, 1, data_size)
retraining_access = np.random.uniform(0, 1, data_size)
WIS = 0.3 * job_displacement + 0.5 * (1 - wage_change) + 0.2 * retraining_access

X_original = np.column_stack((job_displacement, wage_change, retraining_access))
y = WIS

# Create interaction dataset
interaction_terms = (
    job_displacement * wage_change,
    job_displacement * retraining_access,
    wage_change * retraining_access,
)
X_interaction = np.column_stack((X_original, *interaction_terms))

# Train/test split for both datasets
X_train, X_val, y_train, y_val = train_test_split(
    X_original, y, test_size=0.2, random_state=42
)
X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(
    X_interaction, y, test_size=0.2, random_state=42
)


# Dataset class
class WorkerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Create DataLoaders
train_dataset = WorkerDataset(X_train, y_train)
val_dataset = WorkerDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

train_dataset_int = WorkerDataset(X_train_int, y_train_int)
val_dataset_int = WorkerDataset(X_val_int, y_val_int)
train_loader_int = DataLoader(train_dataset_int, batch_size=32, shuffle=True)
val_loader_int = DataLoader(val_dataset_int, batch_size=32)


# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Prepare experiment data storage
experiment_data = {
    "multi_feature_interaction": {
        "original_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "interaction_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# List of learning rates to test
learning_rates = [0.001, 0.01, 0.0001]
num_epochs = 50


# Training and evaluation function
def train_and_evaluate(model, train_loader, val_loader, lr, dataset_key):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            features, targets = batch
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        experiment_data["multi_feature_interaction"][dataset_key]["losses"][
            "train"
        ].append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features, targets = batch
                features, targets = features.to(device), targets.to(device)

                outputs = model(features)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                experiment_data["multi_feature_interaction"][dataset_key][
                    "predictions"
                ].extend(outputs.squeeze().cpu().numpy())
                experiment_data["multi_feature_interaction"][dataset_key][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data["multi_feature_interaction"][dataset_key]["losses"][
            "val"
        ].append(val_loss)

        WIS = 1 - val_loss
        experiment_data["multi_feature_interaction"][dataset_key]["metrics"][
            "val"
        ].append(WIS)

        print(
            f"Dataset = {dataset_key}, Learning Rate = {lr}, Epoch {epoch + 1}: training_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, WIS = {WIS:.4f}"
        )


# Training with original dataset
for lr in learning_rates:
    model = SimpleNN(input_size=3).to(device)
    print(f"Training with original dataset, learning rate: {lr}")
    train_and_evaluate(model, train_loader, val_loader, lr, "original_dataset")

# Training with interaction dataset
for lr in learning_rates:
    model = SimpleNN(input_size=6).to(device)
    print(f"Training with interaction dataset, learning rate: {lr}")
    train_and_evaluate(
        model, train_loader_int, val_loader_int, lr, "interaction_dataset"
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
