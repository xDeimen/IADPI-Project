import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

TRAIN = r"./data/train"
VAL = r"./data/val"

# Define the DenseNet201 model
class CustomDenseNet201(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet201, self).__init__()
        self.model = models.densenet201(weights=None)  # No pre-trained weights
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


num_classes = 2
model = CustomDenseNet201(num_classes)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # DenseNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(TRAIN, transform=transform)
val_dataset = datasets.ImageFolder(VAL, transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Track losses
val_losses = []
train_losses = []

def train(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_acc:.2f}%')
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)  # Record validation loss

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_acc = 100. * correct / total
    print(f'Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_acc:.2f}%')
    return val_loss / len(val_loader)

def plot_validation_loss(val_losses):
    plt.plot(val_losses)
    plt.title("Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def compute_metrics(model, val_loader, device, class_names):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_loader, val_loader, criterion, optimizer, epochs=100, device=device)

    # Save the model
    torch.save(model.state_dict(), 'densenet201_custom.pth')

    # Plot validation loss
    plot_validation_loss(train_losses)

    # Compute confusion matrix and metrics
    class_names = train_dataset.classes
    compute_metrics(model, val_loader, device, class_names)
