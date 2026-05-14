import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic dataset creation
np.random.seed(42)
num_samples = 1000
job_satisfaction = np.random.rand(num_samples)
job_security = np.random.rand(num_samples)
retraining_opportunities = np.random.rand(num_samples)
wwbi = (job_satisfaction + job_security + retraining_opportunities) / 3

X = np.column_stack((job_satisfaction, job_security, retraining_opportunities))
y = wwbi

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define data loaders
def create_data_loader(X, y):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=32)


train_loader = create_data_loader(X_train, y_train)
val_loader = create_data_loader(X_val, y_val)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Function to train and evaluate the model with a specific input set
def train_and_evaluate(X_train, X_val, y_train, y_val):
    model = SimpleNN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    metrics = {"train": [], "val": []}
    losses = {"train": [], "val": []}
    val_predictions, val_targets = [], []

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in create_data_loader(X_train, y_train):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(X_train) * 32  # Normalize
        losses["train"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in create_data_loader(X_val, y_val):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        avg_val_loss = val_loss / len(X_val) * 32  # Normalize
        losses["val"].append(avg_val_loss)
        metrics["val"].append(np.mean(val_predictions))

        print(
            f"Epoch {epoch+1}: train_loss = {avg_train_loss:.4f}, validation_loss = {avg_val_loss:.4f}"
        )

    return metrics, losses, val_predictions, val_targets


# Ablation study
experiment_data = {
    "input_feature_importance": {
        "all_features": {},
        "omit_job_satisfaction": {},
        "omit_job_security": {},
        "omit_retraining_opportunities": {},
    }
}

# Evaluate with all features
(
    experiment_data["input_feature_importance"]["all_features"]["metrics"],
    experiment_data["input_feature_importance"]["all_features"]["losses"],
    experiment_data["input_feature_importance"]["all_features"]["predictions"],
    experiment_data["input_feature_importance"]["all_features"]["ground_truth"],
) = train_and_evaluate(X_train, X_val, y_train, y_val)

# Omit job_satisfaction
(
    experiment_data["input_feature_importance"]["omit_job_satisfaction"]["metrics"],
    experiment_data["input_feature_importance"]["omit_job_satisfaction"]["losses"],
    experiment_data["input_feature_importance"]["omit_job_satisfaction"]["predictions"],
    experiment_data["input_feature_importance"]["omit_job_satisfaction"][
        "ground_truth"
    ],
) = train_and_evaluate(X_train[:, 1:], X_val[:, 1:], y_train, y_val)

# Omit job_security
(
    experiment_data["input_feature_importance"]["omit_job_security"]["metrics"],
    experiment_data["input_feature_importance"]["omit_job_security"]["losses"],
    experiment_data["input_feature_importance"]["omit_job_security"]["predictions"],
    experiment_data["input_feature_importance"]["omit_job_security"]["ground_truth"],
) = train_and_evaluate(X_train[:, [0, 2]], X_val[:, [0, 2]], y_train, y_val)

# Omit retraining_opportunities
(
    experiment_data["input_feature_importance"]["omit_retraining_opportunities"][
        "metrics"
    ],
    experiment_data["input_feature_importance"]["omit_retraining_opportunities"][
        "losses"
    ],
    experiment_data["input_feature_importance"]["omit_retraining_opportunities"][
        "predictions"
    ],
    experiment_data["input_feature_importance"]["omit_retraining_opportunities"][
        "ground_truth"
    ],
) = train_and_evaluate(X_train[:, :2], X_val[:, :2], y_train, y_val)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
