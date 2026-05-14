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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic dataset
np.random.seed(42)
num_samples = 1000
X = np.random.rand(
    num_samples, 4
)  # Features: job retention, satisfaction, fairness, intervention effectiveness
y = (0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * X[:, 2] + 0.1 * X[:, 3] > 0.6).astype(
    float
)  # PWIS as a binary outcome

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


model = SimpleNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize experiment data storage
experiment_data = {
    "synthetic_dataset": {
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
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    experiment_data["synthetic_dataset"]["losses"]["train"].append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            experiment_data["synthetic_dataset"]["predictions"].extend(
                outputs.cpu().numpy()
            )
            experiment_data["synthetic_dataset"]["ground_truth"].extend(
                labels.cpu().numpy()
            )

    avg_val_loss = val_loss / len(val_loader)
    experiment_data["synthetic_dataset"]["losses"]["val"].append(avg_val_loss)

    # Calculate Pro-Worker Impact Score: Proportion of accurate predictions
    pwis = (
        np.array(experiment_data["synthetic_dataset"]["predictions"]) > 0.5
    ) == np.array(experiment_data["synthetic_dataset"]["ground_truth"])
    pwis_score = np.mean(pwis)
    experiment_data["synthetic_dataset"]["metrics"]["train"].append(pwis_score)

    print(
        f"Epoch {epoch + 1}: training_loss = {avg_train_loss:.4f}, validation_loss = {avg_val_loss:.4f}, PWIS = {pwis_score:.4f}"
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
