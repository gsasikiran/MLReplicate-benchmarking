import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import json
import numpy as np
import argparse

# Define the GRU model with a conformal prediction head
class GRUConformalModel(nn.Module):
    def __init__(self, input_dim, gru_units, output_dim):
        super(GRUConformalModel, self).__init__()
        self.gru = nn.GRU(input_dim, gru_units, batch_first=True)
        self.conformal_head = nn.Linear(gru_units, output_dim)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.conformal_head(x[:, -1, :])
        return x

def load_data():
    # Load the MNIST dataset
    dataset = load_dataset('mnist')

    # Normalize and preprocess the data
    def preprocess(batch):
        batch['image'] = [np.array(img, dtype=np.float32) / 255.0 for img in batch['image']]
        return batch

    dataset = dataset.map(preprocess, batched=True)

    # Subsample to the required sizes
    train_data = dataset['train'].select(range(5000))
    validation_data = dataset['test'].select(range(2000))
    test_data = dataset['test'].select(range(2000))

    # Convert to tensors and adjust dimensions
    train_images = torch.tensor(np.stack(train_data['image'])).unsqueeze(1)  # Reshape to (batch_size, 1, 28, 28)
    train_labels = torch.tensor(train_data['label'])
    val_images = torch.tensor(np.stack(validation_data['image'])).unsqueeze(1)  # Reshape to (batch_size, 1, 28, 28)
    val_labels = torch.tensor(validation_data['label'])
    test_images = torch.tensor(np.stack(test_data['image'])).unsqueeze(1)  # Reshape to (batch_size, 1, 28, 28)
    test_labels = torch.tensor(test_data['label'])

    # Create data loaders
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

def build_model():
    # Define the model architecture
    model = GRUConformalModel(input_dim=28, gru_units=64, output_dim=10)
    return model

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(10):  # Increased training to 10 epochs for better learning
        epoch_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.squeeze(1))  # Remove the channel dimension before passing to GRU
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch+1}/10], Loss: {epoch_loss/len(train_loader):.4f}')

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.squeeze(1))  # Remove the channel dimension before passing to GRU
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    metrics = {
        'ConformalPredictionErrorRate': 1 - accuracy
    }
    return metrics

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = load_data()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = build_model().to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, optimizer, criterion, device)

    # Evaluate the model
    metrics = evaluate(model, test_loader, device)
    print(f"Metrics: {metrics}")

    # Save the results
    with open(os.path.join(out_dir, 'final_info.json'), 'w') as f:
        json.dump(metrics, f)

    # Print dataset sizes
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    main(args.out_dir)