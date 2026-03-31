import torch
import torch.nn as nn
import torch.optim as optim
import time

from models.alexnet import AlexNet
from data.dataset import get_dataloaders

# ---------------------------
# Accuracy Function
# ---------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ---------------------------
# Training Function
# ---------------------------
def train(device):

    print("Using device:", device)

    train_loader, test_loader = get_dataloaders(device=device)

    model = AlexNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # lower LR

    epochs = 5

    for epoch in range(epochs):
        model.train()
        start = time.time()

        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(f"[Epoch {epoch+1}] Batch {i}, Loss: {loss.item():.4f}")

        acc = evaluate(model, test_loader, device)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Loss: {running_loss / len(train_loader):.4f}")
        print(f"Accuracy: {acc:.2f}%")
        print(f"Time: {time.time() - start:.2f}s\n")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    train(device)