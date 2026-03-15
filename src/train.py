import os
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np

from dataset import ISICDataset
from model import SkinClassifier
from model_utils import save_model

# 하이퍼파라미터
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 설정
IMAGE_DIR = "D:/Project/Skin_Disease_Classfication/data/ISIC_2019_Training_Input"
CSV_PATH = "D:/Project/Skin_Disease_Classfication/data/ISIC_2019_Training_GroundTruth.csv"
SAVE_PATH = "D:/Project/Skin_Disease_Classfication/checkpoints/best_model_weighted.pt"

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 전체 데이터셋 로드
dataset = ISICDataset(IMAGE_DIR, CSV_PATH, transform=transform)

# 데이터 분할
train_size = int(len(dataset) * TRAIN_RATIO)
val_size = int(len(dataset) * VAL_RATIO)
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 데이터 로더 준비
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("클래스 가중치 계산 중")

train_indices = train_dataset.indices
all_labels = dataset.labels[train_indices] 
train_label_indices = np.argmax(all_labels, axis=1)

label_counts = Counter(train_label_indices)
num_classes = len(label_counts)
total_samples = len(train_label_indices)

class_weights = []
for i in range(num_classes):
    count = label_counts[i]
    weight = total_samples / (num_classes * max(count, 1))
    class_weights.append(weight)

print(f"   -> 계산된 가중치 : {class_weights}")

# 텐서로 변환 + 디바이스에 올리기
weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

# 모델, 손실함수, 옵티마이저 설정
model = SkinClassifier().to(DEVICE)

# CrossEntropyLoss에 가중치 적용
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 학습 루프 시작
best_val_loss = float('inf')
early_stop_counter = 0
patience = 3 

for epoch in range(NUM_EPOCHS):
    print(f"\n⏹ [Epoch {epoch+1}] Start")
    model.train()
    train_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        targets = torch.argmax(labels, dim=1) 

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, targets) 
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        if batch_idx % 100 == 0:
            print(f"   Batch {batch_idx} -> Loss: {loss.item():.4f}")

    train_loss /= len(train_loader.dataset)
    print(f"   [Epoch {epoch+1}] 평균 Train Loss: {train_loss:.4f}")

    # 검증 루프
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 검증 때도 동일하게 라벨 변환 필요
            targets = torch.argmax(labels, dim=1)

            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"[{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 베스트 모델 저장 및 EarlyStopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, SAVE_PATH)
        print(f"   모델 저장 성공: {SAVE_PATH}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"   EarlyStopping 카운터 증가: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("   검증 손실 개선되지 않아 학습 중단")
            break