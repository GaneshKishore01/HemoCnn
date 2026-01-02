import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
model.load_state_dict(torch.load("blood_cells_classifier.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_dataset = datasets.ImageFolder(
    r"E:\image datasets\RBC,WBC PLT\eval_subset",
    transform=transform
)

eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
classes = eval_dataset.classes

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in eval_loader:
        outputs = model(images)
        preds = outputs.argmax(1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\nOverall Accuracy: {accuracy:.4f}")

report = classification_report(
    y_true,
    y_pred,
    target_names=classes,
    digits=4,
    output_dict=True
)

df_report = pd.DataFrame(report).transpose()
print("\nClassification Report:\n")
print(df_report)
df_report.to_csv("classification_report.csv")
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
print("\nConfusion Matrix:\n")
print(df_cm)
df_cm.to_csv("confusion_matrix.csv")
