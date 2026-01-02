import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder("E:\image datasets\RBC,WBC PLT\set", transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = models.mobilenet_v2(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False
num_classes = 8
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

for epoch in range(10):
    total = 0
    correct = 0
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Epoch {epoch+1} - Loss: {running_loss:.4f} - Acc: {acc:.4f}")

torch.save(model.state_dict(), "blood_cells_classifier.pth")
print("Model saved as blood_cells_classifier.pth")
