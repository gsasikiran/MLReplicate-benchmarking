import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from datasets import load_dataset
import numpy as np
import argparse
import json

def load_data():
    # Load the dataset
    dataset = load_dataset('ag_news')
    
    # Subsample the dataset
    train_data = dataset['train'].select(range(3000))
    valid_data = dataset['test'].select(range(1000))
    test_data = dataset['test'].select(range(1000))
    
    # Transform text to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000)
    
    train_texts = [item['text'] for item in train_data]
    valid_texts = [item['text'] for item in valid_data]
    test_texts = [item['text'] for item in test_data]
    
    train_features = vectorizer.fit_transform(train_texts).toarray()
    valid_features = vectorizer.transform(valid_texts).toarray()
    test_features = vectorizer.transform(test_texts).toarray()
    
    # Dummy target values for regression
    train_labels = np.random.rand(3000)
    valid_labels = np.random.rand(1000)
    test_labels = np.random.rand(1000)
    
    # Convert to Tensors
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    valid_dataset = TensorDataset(torch.tensor(valid_features, dtype=torch.float32), torch.tensor(valid_labels, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}, Test size: {len(test_dataset)}")
    return train_loader, valid_loader, test_loader

def build_model():
    # Define a deeper MLP model
    model = nn.Sequential(
        nn.Linear(1000, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return model

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(10):  # Increase epochs for better training
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
        
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    return {"MAE": mae, "R2": r2}

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    train_loader, valid_loader, test_loader = load_data()
    
    # Build model
    model = build_model()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train model
    train(model, train_loader, optimizer, criterion, device)
    
    # Evaluate model
    metrics = evaluate(model, test_loader, device)
    
    # Save metrics
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(metrics, f)
    
    print("Metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    
    main(args.out_dir)