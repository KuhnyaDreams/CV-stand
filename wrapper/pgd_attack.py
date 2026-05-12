import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import os
from model_functions import detect
# 1. Загружаем модель
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

PHONE_CLASS = 737 
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)

def save_adversarial_image(img_tensor, output_path):
    img = img_tensor.detach().squeeze().cpu().numpy().copy()
    
    # 🔑 Ключевое исправление: reshape для broadcasting
    mean_bgr = IMAGENET_MEAN_BGR.reshape(3, 1, 1)
    img += mean_bgr
    
    # BGR -> RGB
    img = img[[2, 1, 0], :, :]
    
    # (C,H,W) -> (H,W,C) для PIL
    img = img.transpose(1, 2, 0)
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)
    print(f"✅ Сохранено: {os.path.abspath(output_path)}")

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    
    # RGB -> BGR + вычитание среднего
    img_array = img_array[:, :, ::-1] - IMAGENET_MEAN_BGR
    
    # .copy() для contiguous memory + transpose к (C,H,W)
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1).copy()).unsqueeze(0)
    return img_tensor

def avoid_class_pgd(model, image_path, avoid_class, epsilon=1.0, max_iter=15):
    original_full = load_and_preprocess_image(image_path)
    h, w = original_full.shape[2], original_full.shape[3]
    print(f"📷 Исходный размер изображения: {w}x{h}")

    original_224 = F.interpolate(original_full, size=(224, 224), mode='bilinear', align_corners=False)
    adv_224 = original_224.clone().requires_grad_(True)

    with torch.no_grad():
        init_logits = model(original_224)
        init_prob = F.softmax(init_logits, dim=1)[0, avoid_class].item()
    print(f"📉 Исходная вероятность 'телефон' (на 224x224): {init_prob:.4f}")

    for i in range(max_iter):
        predictions = model(adv_224)
        loss = F.cross_entropy(predictions, torch.tensor([avoid_class]))

        loss.backward()
        grads = adv_224.grad.data
        adv_224.data = adv_224.data + epsilon * torch.sign(grads)
        adv_224.data = torch.clamp(adv_224.data, -125.0, 155.0)
        adv_224.grad.zero_()

        if i % 3 == 0:
            with torch.no_grad():
                current_prob = F.softmax(model(adv_224), dim=1)[0, avoid_class].item()
            print(f"   Итерация {i+1}: вероятность = {current_prob:.4f}")

    delta_224 = adv_224 - original_224
    delta_full = F.interpolate(delta_224, size=(h, w), mode='bilinear', align_corners=False)
    
    adv_full = (original_full + delta_full).detach()
    adv_full = torch.clamp(adv_full, -125.0, 155.0)

    print(f"✅ Атака завершена. Итоговое изображение: {w}x{h}")
    return adv_full

# --- ЗАПУСК ---
IMAGE_PATH = '../data/test.png'
adversarial_img = avoid_class_pgd(model, IMAGE_PATH, PHONE_CLASS, epsilon=1.0, max_iter=15)
save_adversarial_image(adversarial_img, '../data/adversarial_no_phone.png')
detect(input_path='test.png')
detect(input_path='adversarial_no_phone.png')