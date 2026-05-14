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


# Ablation Study: Varying Input Noise Levels
noise_levels = [
    0.0,
    0.01,
    0.05,
    0.1,
    0.2,
]  # Different noise levels (standard deviations)
batch_size = 32
epochs = 100
experiment_data = {"varying_input_noise": {}}

for noise in noise_levels:
    experiment_data["varying_input_noise"][f"noise_level_{noise}"] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Adding noise to the training and validation sets
    noisy_X_train = X_train + np.random.normal(0, noise, X_train.shape)
    noisy_X_val = X_val + np.random.normal(0, noise, X_val.shape)

    noisy_X_train_tensor = torch.tensor(noisy_X_train, dtype=torch.float32).to(device)
    noisy_X_val_tensor = torch.tensor(noisy_X_val, dtype=torch.float32).to(device)

    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create data loaders
    train_data = torch.utils.data.TensorDataset(noisy_X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    # Training Loop
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

        experiment_data["varying_input_noise"][f"noise_level_{noise}"]["losses"][
            "train"
        ].append(train_loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(noisy_X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)
            experiment_data["varying_input_noise"][f"noise_level_{noise}"]["losses"][
                "val"
            ].append(val_loss.item())

        # Calculate PWIS as a metric (for demonstration)
        PWIS = 1 - val_loss.item()  # Higher is better
        experiment_data["varying_input_noise"][f"noise_level_{noise}"]["metrics"][
            "val"
        ].append(PWIS)

        print(
            f"Noise Level {noise}, Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss.item():.4f}, Validation Loss = {val_loss.item():.4f}, PWIS = {PWIS:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
