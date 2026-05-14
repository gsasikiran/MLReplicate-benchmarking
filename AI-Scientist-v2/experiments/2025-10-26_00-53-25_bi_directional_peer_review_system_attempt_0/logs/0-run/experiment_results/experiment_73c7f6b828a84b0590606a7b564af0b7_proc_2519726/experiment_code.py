import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic data generation
np.random.seed(0)
num_samples = 1000
X = np.random.rand(num_samples, 3)  # Simulated features: clarity, depth, relevance
RQS = np.clip(
    X[:, 0] * 0.5
    + X[:, 1] * 0.3
    + X[:, 2] * 0.2
    + np.random.normal(0, 0.05, num_samples),
    0,
    1,
)

X_train, X_val, y_train, y_val = train_test_split(
    X, RQS, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define the neural network model
class RQSModel(nn.Module):
    def __init__(self, hidden_activation=nn.ReLU(), output_activation=nn.Sigmoid()):
        super(RQSModel, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def forward(self, x):
        x = self.hidden_activation(self.fc1(x))
        x = self.output_activation(self.fc2(x))
        return x


# Hyperparameter tuning: Different activation functions to test
activation_functions = {
    "relu_sigmoid": (nn.ReLU(), nn.Sigmoid()),
    "leaky_relu_sigmoid": (nn.LeakyReLU(), nn.Sigmoid()),
    "tanh_sigmoid": (nn.Tanh(), nn.Sigmoid()),
    "relu_tanh": (nn.ReLU(), nn.Tanh()),
    "leaky_relu_tanh": (nn.LeakyReLU(), nn.Tanh()),
    "tanh_tanh": (nn.Tanh(), nn.Tanh()),
}

experiment_data = {
    "hyperparam_tuning_activation": {
        function_name: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        for function_name in activation_functions
    }
}

# Training loop
num_epochs = 50
for name, (hidden_activation, output_activation) in activation_functions.items():
    model = RQSModel(hidden_activation, output_activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_train_pred = model(X_train_tensor)
        train_loss = criterion(y_train_pred.squeeze(), y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            val_loss = criterion(y_val_pred.squeeze(), y_val_tensor)

        # Update metrics
        experiment_data["hyperparam_tuning_activation"][name]["metrics"][
            "train"
        ].append(1 - train_loss.item())
        experiment_data["hyperparam_tuning_activation"][name]["losses"]["train"].append(
            train_loss.item()
        )
        experiment_data["hyperparam_tuning_activation"][name]["metrics"]["val"].append(
            1 - val_loss.item()
        )
        experiment_data["hyperparam_tuning_activation"][name]["losses"]["val"].append(
            val_loss.item()
        )
        experiment_data["hyperparam_tuning_activation"][name]["predictions"].append(
            y_val_pred.cpu().numpy()
        )
        experiment_data["hyperparam_tuning_activation"][name]["ground_truth"].append(
            y_val_tensor.cpu().numpy()
        )

        print(
            f"{name} - Epoch {epoch+1}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

# Save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
