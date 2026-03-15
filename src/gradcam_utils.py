import cv2
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def show_gradcam(model, image_tensor, class_idx=None, target_layer_name=None, device='cpu'):
    model.eval()

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.cpu().detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].cpu().detach())

    # target layer hook 등록
    target_layer = dict(model.backbone.named_modules())[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)

    if class_idx is None:
        class_idx = torch.argmax(output)
        print("\n" + "="*40)
        print(f"[Grad-CAM 분석 결과]")
        print(f"모델의 예측(Predicted): Class {class_idx.item()}")
        print("="*40 + "\n")

    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    forward_handle.remove()
    backward_handle.remove()

    # gradcam 계산
    grad = gradients[0]
    act = activations[0]

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze(0)

    cam = np.maximum(cam.numpy(), 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    image_np = to_pil_image(image_tensor[0].cpu()).convert('RGB')
    image_np = np.array(image_np)

    overlayed = heatmap * 0.4 + image_np

    plt.imshow(overlayed.astype(np.uint8))
    plt.title(f"Grad-CAM: class {class_idx}")
    plt.axis('off')
    plt.show()
