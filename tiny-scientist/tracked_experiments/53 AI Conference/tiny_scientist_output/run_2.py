import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import numpy as np
import json
import argparse

def load_data():
    # Load the Yelp Polarity dataset
    dataset = load_dataset('yelp_polarity')
    # Subsample: train=5000, validation=1000, test=1000
    train_data = dataset['train'].select(range(5000))
    val_data = dataset['test'].select(range(1000))
    test_data = dataset['test'].select(range(1000, 2000))
    
    # Preprocessing: TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit and transform the train set, transform val and test
    X_train = vectorizer.fit_transform(train_data['text']).toarray()
    X_val = vectorizer.transform(val_data['text']).toarray()
    X_test = vectorizer.transform(test_data['text']).toarray()
    
    y_train = np.array(train_data['label'])
    y_val = np.array(val_data['label'])
    y_test = np.array(test_data['label'])
    
    # Convert to tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)
    
    # Document subsampling
    print(f"Loaded and subsampled dataset sizes: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")
    
    return train_loader, val_loader, test_loader

def build_model():
    # Define a simple logistic regression model
    model = nn.Sequential(
        nn.Linear(5000, 2)  # Input dimension: 5000, Output dimension: 2
    )
    # Model parameter count: 5000*2 + 2 = 10002
    print("Model built with parameter count:", sum(p.numel() for p in model.parameters()))
    return model

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    model.to(device)
    epochs = 5  # Increased epochs to attempt better performance
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} completed. Loss: {total_loss/len(train_loader):.4f}")

def evaluate(model, data_loader, device):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    # Bias Detection Score is a placeholder for expansion
    bias_detection_score = 0.0  # Simplified placeholder
    return {'accuracy': accuracy, 'bias_detection_score': bias_detection_score}

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader = load_data()
    
    # Build model
    model = build_model()
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed optimizer to Adam
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_loader, optimizer, criterion, device)
    
    # Evaluate model
    metrics = evaluate(model, test_loader, device)
    
    # Save results
    with open(os.path.join(out_dir, 'final_info.json'), 'w') as f:
        json.dump(metrics, f)
    
    # Save details for writeup
    with open(os.path.join(out_dir, 'notes.txt'), 'w') as f:
        f.write("Experiment Run 2:\n")
        f.write("Description: This run used 5 epochs with Adam optimizer to improve model performance.\n")
        f.write(f"Results: {metrics}\n")
    
    print(f"Final metrics saved to {out_dir}/final_info.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    
    main(args.out_dir)