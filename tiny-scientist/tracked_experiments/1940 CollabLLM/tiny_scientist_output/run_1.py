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
    # Load dataset using the specified loader
    dataset = load_dataset('imdb')

    # Limit dataset size to 5000 train / 2000 val / 2000 test
    train_data = dataset['train'].shuffle(seed=42).select(range(5000))
    validation_data = dataset['test'].shuffle(seed=42).select(range(2000))
    test_data = dataset['test'].shuffle(seed=42).select(range(2000))

    # Use TF-IDF to convert text to 1000-dimensional vector
    vectorizer = TfidfVectorizer(max_features=1000)

    # Fit and transform train data
    X_train = vectorizer.fit_transform(train_data['text']).toarray()
    y_train = np.array(train_data['label'])

    # Transform validation and test data
    X_val = vectorizer.transform(validation_data['text']).toarray()
    y_val = np.array(validation_data['label'])

    X_test = vectorizer.transform(test_data['text']).toarray()
    y_test = np.array(test_data['label'])

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

    return train_loader, val_loader, test_loader

def build_model():
    # Define a shallow MLP model
    model = nn.Sequential(
        nn.Linear(1000, 64),  # Input layer with 1000 features
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()  # Output layer for binary classification
    )
    # Model has approximately 75,000 parameters
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
        print(f"Epoch {epoch + 1} complete")

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            preds = (outputs > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return {'accuracy': accuracy, 'f1_score': f1}

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_loader, val_loader, test_loader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    train(model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    # Save results
    final_info = {
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    with open(os.path.join(out_dir, 'final_info.json'), 'w') as f:
        json.dump(final_info, f)

    # Print dataset sizes for verification
    print(f"Training size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    main(args.out_dir)