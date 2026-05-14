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

# Synthetic Data Generation
np.random.seed(0)
X = np.random.rand(
    1000, 3
)  # Features: job displacement rate, income stability, empowerment score
y = (
    0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + np.random.normal(0, 0.1, 1000)
)  # PWIS

# Normalize Features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define the models
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=2)  # For 1D input
        self.fc1 = nn.Linear(8 * 2, 10)  # Adjust according to conv output
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten for linear layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=10, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding sequence dimension
        out, _ = self.rnn(x)
        x = self.fc(out[:, -1, :])  # Take the last output
        return x


# Ablation study
experiment_data = {
    "different_model_architectures": {
        "SimpleNN": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "CNN": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "RNN": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# Common training parameters
epochs = 100
batch_size = 32
criterion = nn.MSELoss()
optimizer_type = optim.Adam
learning_rate = 0.001


# Training function
def train_model(model, model_name):
    model.to(device)

    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

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

        experiment_data["different_model_architectures"][model_name]["losses"][
            "train"
        ].append(train_loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            experiment_data["different_model_architectures"][model_name]["losses"][
                "val"
            ].append(val_loss.item())

        # Calculate PWIS as a metric (for demonstration)
        PWIS = 1 - val_loss.item()  # Higher is better
        experiment_data["different_model_architectures"][model_name]["metrics"][
            "val"
        ].append(PWIS)

        # Store predictions and ground truth
        experiment_data["different_model_architectures"][model_name][
            "predictions"
        ].append(val_outputs.cpu().numpy())
        experiment_data["different_model_architectures"][model_name][
            "ground_truth"
        ].append(y_val_tensor.cpu().numpy())

        print(
            f"{model_name} - Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss.item():.4f}, Validation Loss = {val_loss.item():.4f}, PWIS = {PWIS:.4f}"
        )


# Train all models
models = {"SimpleNN": SimpleNN(), "CNN": CNN(), "RNN": RNN()}
for model_name, model in models.items():
    train_model(model, model_name)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
