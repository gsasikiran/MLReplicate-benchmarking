import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic dataset
np.random.seed(42)
num_samples = 1000
X = np.random.rand(num_samples, 3)  # Features: clarity, depth, relevance
y = np.random.rand(num_samples)  # Review Quality Score

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
train_data = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32).to(device),
    torch.tensor(y_train, dtype=torch.float32).to(device),
)
val_data = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32).to(device),
    torch.tensor(y_val, dtype=torch.float32).to(device),
)

# Create DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# Define a simple neural network model
class ReviewQualityModel(nn.Module):
    def __init__(self):
        super(ReviewQualityModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Initialize model, loss function, and optimizer
model = ReviewQualityModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data structure to store experiment data
experiment_data = {
    "synthetic_reviews": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        features, labels = batch
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    experiment_data["synthetic_reviews"]["losses"]["train"].append(avg_train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    rqs_scores = []
    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch
            outputs = model(features)
            loss = criterion(outputs, labels.view(-1, 1))
            val_loss += loss.item()
            rqs_scores.extend(outputs.cpu().numpy().flatten().tolist())

    avg_val_loss = val_loss / len(val_loader)
    experiment_data["synthetic_reviews"]["losses"]["val"].append(avg_val_loss)
    experiment_data["synthetic_reviews"]["metrics"]["val"].append(
        np.mean(rqs_scores)
    )  # RQS Measurement

    print(
        f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, validation_loss = {avg_val_loss:.4f}"
    )

# Save metrics and losses
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
