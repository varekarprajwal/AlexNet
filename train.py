import torch
import torch.nn as nn
import torch.optim as optim
import time

from models.alexnet import AlexNet
from data.dataset import get_dataloaders


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def train(device):

    model = AlexNet(num_classes=10).to(device)
    train_loader, test_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3

    for epoch in range(epochs):
        start = time.time()

        model.train()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch+1}] Batch {i}, Loss: {loss.item():.4f}")

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1} done in {time.time() - start:.2f}s | Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(device)
    train(device)