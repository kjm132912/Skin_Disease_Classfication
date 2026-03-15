# gradcam.py
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SkinClassifier
from model_utils import load_model
from gradcam_utils import show_gradcam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "D:/Project/Skin_Disease_Classfication/checkpoints/best_model_weighted.pt"
IMAGE_PATH = "D:/Project/Skin_Disease_Classfication/data/ISIC_2019_Training_Input/ISIC_0072267.jpg"

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).to(DEVICE)

# 모델 로드
model = load_model(SkinClassifier, CHECKPOINT_PATH, device=DEVICE)

# Grad-CAM 실행
show_gradcam(model, input_tensor, target_layer_name="layer4", device=DEVICE)
