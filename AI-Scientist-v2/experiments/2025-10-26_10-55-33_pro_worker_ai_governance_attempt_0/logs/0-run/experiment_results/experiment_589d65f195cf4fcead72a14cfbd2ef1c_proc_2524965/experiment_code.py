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
X = np.random.rand(1000, 3)
y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + np.random.normal(0, 0.1, 1000)

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


# Define the model
class SimpleNN(nn.Module):
    def __init__(self, init_type="default"):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        if init_type == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif init_type == "he":
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        elif init_type == "uniform":
            nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
            nn.init.uniform_(self.fc2.weight, -0.1, 0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Ablation Study for Initialization Methods
initialization_methods = ["default", "xavier", "he", "uniform"]
epochs = 100
experiment_data = {"varying_model_initialization": {}}

for init_method in initialization_methods:
    experiment_data["varying_model_initialization"][init_method] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    model = SimpleNN(init_type=init_method).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

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

        experiment_data["varying_model_initialization"][init_method]["losses"][
            "train"
        ].append(train_loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            experiment_data["varying_model_initialization"][init_method]["losses"][
                "val"
            ].append(val_loss.item())

        # Calculate and record performance metric
        PWIS = 1 - val_loss.item()  # Higher is better
        experiment_data["varying_model_initialization"][init_method]["metrics"][
            "val"
        ].append(PWIS)

        print(
            f"Initialization: {init_method}, Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss.item():.4f}, Validation Loss = {val_loss.item():.4f}, PWIS = {PWIS:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
