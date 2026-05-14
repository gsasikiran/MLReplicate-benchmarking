import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic dataset creation
np.random.seed(42)
num_samples = 1000
X = np.random.rand(num_samples, 5)  # Features representing reviews
y = np.random.rand(num_samples, 1)  # Author ratings

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Model definition
class ReviewerQualityModel(nn.Module):
    def __init__(self):
        super(ReviewerQualityModel, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training setup
model = ReviewerQualityModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data preparation
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# Experiment data initialization
experiment_data = {
    "peer_review_quality": {
        "metrics": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    # RQI Calculation (hypothetical)
    RQI = 1 - val_loss.item()  # Simple transformation for illustration
    experiment_data["peer_review_quality"]["metrics"]["train"].append(loss.item())
    experiment_data["peer_review_quality"]["metrics"]["val"].append(val_loss.item())
    experiment_data["peer_review_quality"]["predictions"].extend(
        val_outputs.cpu().numpy().flatten().tolist()
    )
    experiment_data["peer_review_quality"]["ground_truth"].extend(
        y_val.flatten().tolist()
    )

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, RQI = {RQI:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
