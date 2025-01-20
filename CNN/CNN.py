import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.01

# Custom dataset class
class TumorDataset(Dataset):
    def __init__(self, healthy_path, tumor_path, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        print("Loading data...")
        for path, label in [(healthy_path, 0), (tumor_path, 1)]:
            for file in os.listdir(path):
                img_path = os.path.join(path, file)
                if not os.path.isfile(img_path):
                    continue
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                self.data.append(img)
                self.labels.append(label)

        print(f"Data loaded: {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_nr_of_tumor(self):
        sum = 0
        for x in self.labels:
            if x == 1:
                sum= sum +1
        return sum

    def get_nr_of_clean(self):
        sum = 0
        for x in self.labels:
            if x == 0:
                sum= sum +1
        return sum
            

# Model definition (simple ResNet-inspired CNN)
class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 28 * 28, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    model.to(device)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Training Loss: {running_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    from torchvision import transforms

    healthy_path = r'./data/non_tum'
    tumor_path = r'./data/tum'

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    dataset = TumorDataset(healthy_path, tumor_path, transform=transform)
    print(dataset.get_nr_of_clean())
    print(dataset.get_nr_of_tumor())
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleResNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device)

    print("Training completed.")
