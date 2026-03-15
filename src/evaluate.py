# src/evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import SkinClassifier                   
from model_utils import load_model                
from dataset import ISICDataset                   
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 설정 
IMAGE_DIR = "D:/Project/Skin_Disease_Classfication/data/ISIC_2019_Training_Input"
CSV_PATH = "D:/Project/Skin_Disease_Classfication/data/ISIC_2019_Training_GroundTruth.csv"

CHECKPOINT_PATH = "D:/Project/Skin_Disease_Classfication/checkpoints/best_model_weighted.pt"
BATCH_SIZE = 32
NUM_CLASSES = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== 전처리 정의 ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])

# 테스트셋,  로딩
test_dataset = ISICDataset(IMAGE_DIR, CSV_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = load_model(SkinClassifier, CHECKPOINT_PATH, device=DEVICE)
criterion = nn.CrossEntropyLoss()

# 평가 시작
model.eval()
running_loss = 0.0
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        labels = torch.argmax(labels, dim=1)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

avg_loss = running_loss / len(test_loader.dataset)
print(f"\n Test Loss : {avg_loss:.4f}\n")

# 평가 지표 출력
print("Classification Report")
print(classification_report(true_labels, pred_labels))

# Confusion Matrix 출력
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC', 'SCC']
plot_confusion_matrix(true_labels, pred_labels, classes=class_names)
