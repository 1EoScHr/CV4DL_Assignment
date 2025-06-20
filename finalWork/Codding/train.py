import torch
import torch.nn as nn
import torch.optim as optim
from dataload import get_dataLoader
from model import AlexNet, LSTM, ViT  
from utils import evaluate_accuracy  

def train(model, train_loader, test_loader, epochs=10, lr=1e-3, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate_accuracy(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Acc: {acc:.2f}%")
