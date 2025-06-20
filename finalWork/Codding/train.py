import torch
import torch.nn as nn
import torch.optim as optim
import time

from utils import evaluate_accuracy  

def train(model_name, model, train_loader, val_loader, test_loader, epochs=10, lr=1e-3, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if torch.cuda.is_available():   # GPU上异步并行，需要同步时钟
        torch.cuda.synchronize()

    start_time = time.time()        # 获取时间

    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate_accuracy(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Acc: {acc:.2f}%")

        if acc > best_acc:      # 记录表现最好的
            best_acc = acc
            best_model_state = model.state_dict()
            print("--- best model renew ---")

    model.load_state_dict(best_model_state)
    end_time = time.time()

    print(f"best model TEST acc: {evaluate_accuracy(model, test_loader, device):.2f}%, total time: {end_time - start_time:.2f} s")

    pth = model_name + "_best.pth"
    torch.save(best_model_state, pth)
    print("model state have been saved at " + pth)
