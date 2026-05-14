import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 1)
y = 3 * X.squeeze() + np.random.normal(0, 0.5, X.shape[0])

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define a simple Bayesian linear regression model
class BayesianLinearRegression(nn.Module):
    def __init__(self):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Experiment data
experiment_data = {
    "ablation_learning_rate_scheduler": {
        "synthetic_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
    "baseline": {
        "synthetic_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    },
}


# Function to train the model without learning rate scheduler
def train_without_scheduler(model, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        predictions = model(X_train_tensor)
        train_loss = criterion(predictions.squeeze(), y_train_tensor)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions.squeeze(), y_val_tensor)

        # Save metrics
        experiment_data["baseline"]["synthetic_data"]["losses"]["train"].append(
            train_loss.item()
        )
        experiment_data["baseline"]["synthetic_data"]["losses"]["val"].append(
            val_loss.item()
        )
        experiment_data["baseline"]["synthetic_data"]["predictions"].append(
            val_predictions.cpu().numpy()
        )
        experiment_data["baseline"]["synthetic_data"]["ground_truth"].append(y_val)

        print(
            f"[Baseline] Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )


# Function to train the model with learning rate scheduler
def train_with_scheduler(model, optimizer, scheduler, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        predictions = model(X_train_tensor)
        train_loss = criterion(predictions.squeeze(), y_train_tensor)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions.squeeze(), y_val_tensor)

        # Save metrics
        experiment_data["ablation_learning_rate_scheduler"]["synthetic_data"]["losses"][
            "train"
        ].append(train_loss.item())
        experiment_data["ablation_learning_rate_scheduler"]["synthetic_data"]["losses"][
            "val"
        ].append(val_loss.item())
        experiment_data["ablation_learning_rate_scheduler"]["synthetic_data"][
            "predictions"
        ].append(val_predictions.cpu().numpy())
        experiment_data["ablation_learning_rate_scheduler"]["synthetic_data"][
            "ground_truth"
        ].append(y_val)

        print(
            f"[Scheduler] Epoch {epoch}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )


# Training configurations
epochs = 100
momentum = 0.9
lr = 0.01

# Baseline training without scheduler
print("Training Baseline Model without Scheduler:")
model_baseline = BayesianLinearRegression().to(device)
criterion = nn.MSELoss()
optimizer_baseline = optim.SGD(model_baseline.parameters(), lr=lr, momentum=momentum)
train_without_scheduler(model_baseline, optimizer_baseline, criterion, epochs)

# Training with learning rate scheduler
print("\nTraining Model with Learning Rate Scheduler:")
model_scheduler = BayesianLinearRegression().to(device)
optimizer_scheduler = optim.SGD(model_scheduler.parameters(), lr=lr, momentum=momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer_scheduler, step_size=20, gamma=0.1)
train_with_scheduler(model_scheduler, optimizer_scheduler, scheduler, criterion, epochs)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
