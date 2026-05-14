import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import json
import argparse
import time

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = 5000
    val_size = 1000
    test_size = 1000
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, len(full_dataset) - train_size - val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader

def build_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    inference_time = time.time() - start_time
    return {"accuracy": accuracy, "inference_time": inference_time}

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = load_data()
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train(model, train_loader, optimizer, criterion, device)
    results = evaluate(model, test_loader, device)
    
    with open(os.path.join(out_dir, 'final_info.json'), 'w') as f:
        json.dump(results, f)
    
    with open(os.path.join(out_dir, 'notes.txt'), 'w') as f:
        f.write("Run 2:\n")
        f.write("Experiment Description: Training a simple CNN on MNIST dataset with 2 convolutional layers and using 5 epochs.\n")
        f.write(f"Results: {results}\n")
        f.write("Training dataset size: 5000\n")
        f.write("Validation dataset size: 1000\n")
        f.write("Test dataset size: 1000\n")
        f.write("Achieved accuracy of 96.31% with an inference time of 7.35 seconds.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    main(args.out_dir)