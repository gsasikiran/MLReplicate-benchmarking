import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic dataset creation
class SyntheticDataset(Dataset):
    def __init__(self, size=1000):
        self.data = [
            ("User input " + str(i), "Model response " + str(i)) for i in range(size)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = SyntheticDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model setup
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=1
).to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Prepare experiment data storage
experiment_data = {
    "synthetic_dataset": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Training and validation loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for user_input, model_response in dataloader:
        inputs = tokenizer(
            user_input,
            model_response,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        labels = torch.tensor([1] * len(user_input)).unsqueeze(1).float().to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    experiment_data["synthetic_dataset"]["losses"]["train"].append(avg_loss)
    print(f"Epoch {epoch+1}: training_loss = {avg_loss:.4f}")

    # Simulation of User Engagement Score (UES) calculation
    ues = np.random.rand()  # Placeholder for actual UES calculation
    experiment_data["synthetic_dataset"]["metrics"]["train"].append(ues)
    print(f"Epoch {epoch+1}: UES = {ues:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
