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
    def __init__(self, input_size):
        super(RQSModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Function to train and evaluate the model
def train_and_evaluate(
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epochs=50, lr=0.001
):
    input_size = X_train_tensor.shape[1]  # Use the input size dynamically
    model = RQSModel(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    metrics = {"train": [], "val": []}
    losses = {"train": [], "val": []}
    predictions = []
    ground_truth = []

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
        metrics["train"].append(1 - train_loss.item())
        losses["train"].append(train_loss.item())
        metrics["val"].append(1 - val_loss.item())
        losses["val"].append(val_loss.item())
        predictions.append(y_val_pred.cpu().numpy())
        ground_truth.append(y_val_tensor.cpu().numpy())

        print(
            f"Epoch {epoch+1}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

    return metrics, losses, predictions, ground_truth


# Experiment Data Structure for Ablation Study
experiment_data = {
    "feature_influence_ablation": {
        "full": {},
        "no_clarity": {},
        "no_depth": {},
        "no_relevance": {},
    }
}

# Training with all features
print("\nTraining with all features")
(
    experiment_data["feature_influence_ablation"]["full"]["metrics"],
    experiment_data["feature_influence_ablation"]["full"]["losses"],
    experiment_data["feature_influence_ablation"]["full"]["predictions"],
    experiment_data["feature_influence_ablation"]["full"]["ground_truth"],
) = train_and_evaluate(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

# Ablation: Removing clarity
print("\nTraining without clarity")
X_train_no_clarity = X_train_tensor[:, 1:]  # Remove clarity feature
X_val_no_clarity = X_val_tensor[:, 1:]
(
    experiment_data["feature_influence_ablation"]["no_clarity"]["metrics"],
    experiment_data["feature_influence_ablation"]["no_clarity"]["losses"],
    experiment_data["feature_influence_ablation"]["no_clarity"]["predictions"],
    experiment_data["feature_influence_ablation"]["no_clarity"]["ground_truth"],
) = train_and_evaluate(
    X_train_no_clarity, y_train_tensor, X_val_no_clarity, y_val_tensor
)

# Ablation: Removing depth
print("\nTraining without depth")
X_train_no_depth = X_train_tensor[:, [0, 2]]  # Remove depth feature
X_val_no_depth = X_val_tensor[:, [0, 2]]
(
    experiment_data["feature_influence_ablation"]["no_depth"]["metrics"],
    experiment_data["feature_influence_ablation"]["no_depth"]["losses"],
    experiment_data["feature_influence_ablation"]["no_depth"]["predictions"],
    experiment_data["feature_influence_ablation"]["no_depth"]["ground_truth"],
) = train_and_evaluate(X_train_no_depth, y_train_tensor, X_val_no_depth, y_val_tensor)

# Ablation: Removing relevance
print("\nTraining without relevance")
X_train_no_relevance = X_train_tensor[:, :2]  # Remove relevance feature
X_val_no_relevance = X_val_tensor[:, :2]
(
    experiment_data["feature_influence_ablation"]["no_relevance"]["metrics"],
    experiment_data["feature_influence_ablation"]["no_relevance"]["losses"],
    experiment_data["feature_influence_ablation"]["no_relevance"]["predictions"],
    experiment_data["feature_influence_ablation"]["no_relevance"]["ground_truth"],
) = train_and_evaluate(
    X_train_no_relevance, y_train_tensor, X_val_no_relevance, y_val_tensor
)

# Save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
