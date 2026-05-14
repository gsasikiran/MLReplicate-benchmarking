import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic Dataset Generation
class ReviewDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.X = np.random.rand(num_samples, 3)  # 3 features: clarity, depth, relevance
        self.y = (self.X[:, 0] + self.X[:, 1] + self.X[:, 2]) / 3  # RQS calculation
        self.y += np.random.normal(0, 0.1, num_samples)  # Adding noise

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.X[idx], dtype=torch.float32).to(device),
            "label": torch.tensor(self.y[idx], dtype=torch.float32).to(device),
        }


# Model Definition
class RQSNet(nn.Module):
    def __init__(self):
        super(RQSNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Training and Evaluation
def train_and_evaluate():
    dataset = ReviewDataset()
    train_data, val_data = train_test_split(dataset, test_size=0.2)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = RQSNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    experiment_data = {
        "synthetic_reviews": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
        }
    }

    for epoch in range(100):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch["features"])
            loss = criterion(outputs.view(-1), batch["label"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        experiment_data["synthetic_reviews"]["losses"]["train"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["features"])
                loss = criterion(outputs.view(-1), batch["label"])
                val_loss += loss.item()
        val_loss /= len(val_loader)
        experiment_data["synthetic_reviews"]["losses"]["val"].append(val_loss)

        print(
            f"Epoch {epoch + 1}: training_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


train_and_evaluate()
