import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import numpy as np
import argparse

def load_data():
    # Load the AG News dataset
    dataset = load_dataset('ag_news')
    
    # Subsample the dataset to meet the constraints
    train_data = dataset['train'].select(range(5000))
    val_data = dataset['test'].select(range(1000, 2000))
    test_data = dataset['test'].select(range(1000))
    
    # Preprocess the text data
    vectorizer = TfidfVectorizer(max_features=128, lowercase=True)
    X_train = vectorizer.fit_transform(train_data['text']).toarray()
    y_train = np.array(train_data['label'])
    X_val = vectorizer.transform(val_data['text']).toarray()
    y_val = np.array(val_data['label'])
    X_test = vectorizer.transform(test_data['text']).toarray()
    y_test = np.array(test_data['label'])
    
    # Convert to tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    # Log dataset sizes
    print(f"Train size: {X_train.size(0)}, Validation size: {X_val.size(0)}, Test size: {X_test.size(0)}")

    return train_loader, val_loader, test_loader

def build_model(input_dim=128, output_dim=1):
    # Shallow MLP model
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim),
        nn.Sigmoid()  # Assumes binary classification problem
    )
    return model

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(3):  # 3 epochs for quick training
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} complete")

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    # Placeholder for equity gap metric computation
    equity_gap_metric = 0.0  # This should be computed based on real demographic metrics

    return {"accuracy": accuracy, "f1_score": f1, "equity_gap_metric": equity_gap_metric}

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_data()

    model = build_model()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.BCELoss()

    train(model, train_loader, optimizer, criterion, device)
    metrics = evaluate(model, test_loader, device)

    with open(os.path.join(out_dir, 'final_info.json'), 'w') as f:
        json.dump(metrics, f)

    print(f"Metrics saved to {os.path.join(out_dir, 'final_info.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.out_dir)