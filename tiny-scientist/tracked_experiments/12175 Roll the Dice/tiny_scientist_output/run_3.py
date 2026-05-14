import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
import numpy as np
import argparse

def load_data():
    dataset = load_dataset('ag_news')
    
    # Subsample the dataset
    train_data = dataset['train'].select(range(5000))
    val_data = dataset['test'].select(range(1000))
    test_data = dataset['test'].select(range(1000, 2000))
    
    # Preprocess and convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000, lowercase=True)
    
    # Fit on training data
    train_features = vectorizer.fit_transform([x['text'] for x in train_data]).toarray()
    val_features = vectorizer.transform([x['text'] for x in val_data]).toarray()
    test_features = vectorizer.transform([x['text'] for x in test_data]).toarray()
    
    # Convert labels to tensors
    train_labels = torch.tensor([x['label'] for x in train_data], dtype=torch.long)
    val_labels = torch.tensor([x['label'] for x in val_data], dtype=torch.long)
    test_labels = torch.tensor([x['label'] for x in test_data], dtype=torch.long)
    
    # Wrap in TensorDatasets
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32), train_labels)
    val_dataset = TensorDataset(torch.tensor(val_features, dtype=torch.float32), val_labels)
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32), test_labels)
    
    return train_dataset, val_dataset, test_dataset

def build_model():
    # Model architecture: Shallow MLP with 2 hidden layers
    model = nn.Sequential(
        nn.Linear(1000, 64),   # Input layer with 1000 features, first hidden layer
        nn.ReLU(),
        nn.Linear(64, 32),     # Second hidden layer
        nn.ReLU(),
        nn.Linear(32, 4)       # Output layer for 4 classes
    )
    return model

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(3):  # 3 epochs for quick training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate(model, data_loader, device):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    
    # Calculate primary metric: Entropy
    avg_entropy = np.mean([entropy(output) for output in all_outputs])
    
    # Calculate secondary metric: Novelty score based on Jaccard similarity
    # Convert outputs to boolean for Jaccard computation
    binary_outputs = np.where(all_outputs > 0.5, 1, 0)
    novelty_scores = 1 - pairwise_distances(binary_outputs, metric="jaccard")
    avg_novelty = np.mean(novelty_scores[np.triu_indices_from(novelty_scores, k=1)])

    metrics = {
        "avg_entropy": float(avg_entropy),  # Ensure this is a float
        "avg_novelty": float(avg_novelty)   # Ensure this is a float
    }
    return metrics

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    train_dataset, val_dataset, test_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train(model, train_loader, optimizer, criterion, device)
    metrics = evaluate(model, val_loader, device)
    
    # Save metrics
    with open(os.path.join(out_dir, 'final_info.json'), 'w') as f:
        json.dump(metrics, f)
    
    # Save experiment notes for future writeup
    notes = {
        "experiment_description": "Run 2: Evaluated model entropy and novelty with altered architecture and training parameters.",
        "run_number": 2,
        "metrics": metrics
    }
    with open(os.path.join(out_dir, 'notes.txt'), 'w') as f:
        json.dump(notes, f, indent=4)
    
    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    main(args.out_dir)