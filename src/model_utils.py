import torch
import os

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"모델 저장 성공: {path}")


def load_model(model_class, path, device='cpu'):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"모델 로드 성공: {path}")
    return model
