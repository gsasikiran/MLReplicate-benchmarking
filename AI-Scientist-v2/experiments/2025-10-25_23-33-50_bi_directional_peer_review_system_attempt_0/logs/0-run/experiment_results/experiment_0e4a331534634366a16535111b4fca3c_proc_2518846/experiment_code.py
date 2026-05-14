import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Setting up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleNN(nn.Module):
    def __init__(self, input_size=2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Generate synthetic dataset
class PeerReviewDataset(Dataset):
    def __init__(self, size, features):
        self.data = np.random.rand(
            size, 2
        )  # Two features: author ratings and review scores
        self.labels = np.random.rand(size)  # RQI labels
        self.features = features  # Selected features: 0 for only ratings, 1 for only scores, 2 for both

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.features == 0:  # Only author ratings
            x = x[0:1]  # Keep it 2D shape for the model
        elif self.features == 1:  # Only review scores
            x = x[1:2]  # Keep it 2D shape for the model
        return {
            "features": torch.tensor(x, dtype=torch.float32).to(device),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32).to(device),
        }


# Hyperparameter tuning for learning rates
learning_rates = [0.001, 0.01, 0.1]
batch_size = 64  # Static batch size for simplicity
epochs = 10
experiment_data = {
    "ablation_study": {
        "only_author_ratings": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "only_review_scores": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "both_features": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

for feature_set in range(3):  # 0, 1, 2 for different feature sets
    print(f"Training with feature set: {feature_set}")
    dataset = PeerReviewDataset(size=1000, features=feature_set)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleNN(input_size=1 if feature_set != 2 else 2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                features, labels = batch["features"], batch["label"]
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            experiment_data["ablation_study"][
                (
                    "only_author_ratings"
                    if feature_set == 0
                    else "only_review_scores" if feature_set == 1 else "both_features"
                )
            ]["losses"]["train"].append(avg_train_loss)

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    features, labels = batch["features"], batch["label"]
                    outputs = model(features).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            experiment_data["ablation_study"][
                (
                    "only_author_ratings"
                    if feature_set == 0
                    else "only_review_scores" if feature_set == 1 else "both_features"
                )
            ]["losses"]["val"].append(avg_val_loss)

            # Simulate Review Quality Indicator (RQI)
            rqi = 1 - avg_val_loss
            experiment_data["ablation_study"][
                (
                    "only_author_ratings"
                    if feature_set == 0
                    else "only_review_scores" if feature_set == 1 else "both_features"
                )
            ]["metrics"]["train"].append(rqi)

            print(
                f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}, RQI = {rqi:.4f}"
            )

        # Collect predictions and ground truths
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                features, labels = batch["features"], batch["label"]
                outputs = model(features).squeeze()
                experiment_data["ablation_study"][
                    (
                        "only_author_ratings"
                        if feature_set == 0
                        else (
                            "only_review_scores"
                            if feature_set == 1
                            else "both_features"
                        )
                    )
                ]["predictions"].extend(outputs.cpu().numpy())
                experiment_data["ablation_study"][
                    (
                        "only_author_ratings"
                        if feature_set == 0
                        else (
                            "only_review_scores"
                            if feature_set == 1
                            else "both_features"
                        )
                    )
                ]["ground_truth"].extend(labels.cpu().numpy())

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
