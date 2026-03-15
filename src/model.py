import torch
import torch.nn as nn
import torchvision.models as models

class SkinClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(SkinClassifier, self).__init__()

        # ResNet50 모델 - 기존 출력층을 새로운 분류기 레이어로 교체
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

#모델 저장 or 로드
from model import SkinClassifier
from model_utils import save_model, load_model

if __name__ == "__main__":
    model = SkinClassifier()
    
    # 저장 경로
    path = "D:/Project/Skin_Disease_Classfication/checkpoints/skin_model.pt"
    
    # 저장
    save_model(model, path)

    # 로드
    loaded_model = load_model(SkinClassifier, path)

def get_model(num_classes=8):
    return SkinClassifier(num_classes=num_classes)