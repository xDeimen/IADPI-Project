import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

TEST = r"./data/test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def validate_with_metrics(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_acc = 100. * correct / total
    val_loss_avg = val_loss / len(val_loader)

    # Confusion Matrix and Metrics
    cm = confusion_matrix(all_targets, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Calculate metrics
    report = classification_report(all_targets, all_predictions, target_names=val_loader.dataset.classes)
    print("Classification Report:")
    print(report)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_loader.dataset.classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

    print(f'Validation Loss: {val_loss_avg:.4f}, Accuracy: {val_acc:.2f}%')

if __name__ == "__main__":
    num_classes = 2
    criterion = nn.CrossEntropyLoss()

    model = CustomEfficientNetB0(num_classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(TEST, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.load_state_dict(torch.load('efficientnet_b0_custom.pth'))
    model.to(device)

    validate_with_metrics(model, test_loader, criterion, device)