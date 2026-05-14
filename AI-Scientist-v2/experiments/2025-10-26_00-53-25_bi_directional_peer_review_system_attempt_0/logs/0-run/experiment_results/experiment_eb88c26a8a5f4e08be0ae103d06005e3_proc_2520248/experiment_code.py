import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load datasets from HuggingFace
datasets_ag_news = load_dataset("ag_news", split="train[:2000]").to_pandas()
datasets_imdb = load_dataset("imdb", split="train[:2000]").to_pandas()
datasets_yelp = load_dataset("yelp_polarity", split="train[:2000]").to_pandas()

# Combine datasets
combined_data = np.concatenate(
    [
        datasets_ag_news["text"].values,
        datasets_imdb["text"].values,
        datasets_yelp["text"].values,
    ]
)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
combined_data_vectorized = vectorizer.fit_transform(combined_data).toarray()

# Normalize the input data
combined_data_vectorized = (
    combined_data_vectorized - np.mean(combined_data_vectorized, axis=0)
) / np.std(combined_data_vectorized, axis=0)

# Simulated author ratings for quality (mock data)
RQS = np.random.rand(combined_data_vectorized.shape[0])

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    combined_data_vectorized, RQS, test_size=0.2, random_state=42
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define a more complex model
class RQSModel(nn.Module):
    def __init__(self):
        super(RQSModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 50)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(self.dropout(torch.relu(self.fc2(x)))))
        return x


# Hyperparameter tuning: testing different learning rates
learning_rates = [0.0001, 0.001, 0.01]
experiment_data = {
    "hyperparam_tuning_lr": {
        "RQS": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

num_epochs = 50
for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")

    model = RQSModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_train_pred = model(X_train_tensor).squeeze()
        train_loss = criterion(y_train_pred, y_train_tensor)

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor).squeeze()
            val_loss = criterion(y_val_pred, y_val_tensor)

        # Update metrics
        experiment_data["hyperparam_tuning_lr"]["RQS"]["metrics"]["train"].append(
            1 - train_loss.item()
        )
        experiment_data["hyperparam_tuning_lr"]["RQS"]["losses"]["train"].append(
            train_loss.item()
        )
        experiment_data["hyperparam_tuning_lr"]["RQS"]["metrics"]["val"].append(
            1 - val_loss.item()
        )
        experiment_data["hyperparam_tuning_lr"]["RQS"]["losses"]["val"].append(
            val_loss.item()
        )
        experiment_data["hyperparam_tuning_lr"]["RQS"]["predictions"].append(
            y_val_pred.cpu().numpy()
        )
        experiment_data["hyperparam_tuning_lr"]["RQS"]["ground_truth"].append(
            y_val_tensor.cpu().numpy()
        )

        print(
            f"Epoch {epoch + 1}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

# Save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
