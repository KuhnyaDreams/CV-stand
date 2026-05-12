import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import numpy as np
from PIL import Image
import os
from model_functions import detect

# Загружаем предобученную ResNet50
model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()  # Режим инференса

# Класс "cellular telephone" в ImageNet (индекс 737)
PHONE_CLASS = 737

# Параметры нормализации ImageNet
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def save_adversarial_image(img_tensor, output_path):
    """
    Сохраняет изображение, возвращая его из нормализованного формата в [0, 255]
    """
    img = img_tensor[0].detach().cpu().numpy().copy()
    
    # C, H, W -> H, W, C
    img = np.transpose(img, (1, 2, 0))
    
    # Денормализация: img = img * std + mean
    img = img * IMAGENET_STD + IMAGENET_MEAN
    
    # Конвертация в [0, 255]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # Сохранение через PIL (ожидает RGB)
    Image.fromarray(img).save(output_path)
    print(f"✅ Сохранено: {os.path.abspath(output_path)}")


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Загружает и препроцессит изображение для модели
    Возвращает: тензор (1, 3, H, W) и оригинальный размер
    """
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size  # (width, height)
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), 
                           std=IMAGENET_STD.tolist())
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Добавляем batch dimension
    return img_tensor, orig_size


def load_original_image(image_path):
    """
    Загружает изображение в исходном размере без ресайза
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    return img_array, img.size


def deepfool_avoid_class(model, image, avoid_class, num_classes=10, 
                         overshoot=0.02, max_iter=50):
    """
    DeepFool атака для избегания определённого класса.
    
    Алгоритм ищет минимальное возмущение, чтобы изображение перестало 
    классифицироваться как avoid_class, пересекая ближайшую границу решения.
    
    Args:
        model: PyTorch модель в eval mode
        image: входной тензор (1, 3, 224, 224), нормализованный
        avoid_class: индекс класса, которого нужно избежать
        num_classes: количество соседних классов для поиска границы
        overshoot: коэффициент перерегулирования (обычно 0.02)
        max_iter: максимальное число итераций
    
    Returns:
        adv_image: возмущённое изображение (тензор)
        n_iter: фактическое число итераций
    """
    model.eval()
    image = image.clone().detach()
    
    # Проверка: если изображение уже не классифицируется как avoid_class
    with torch.no_grad():
        output = model(image)
    
    if torch.argmax(output).item() != avoid_class:
        print(f"⚠️ Изображение уже не классифицируется как класс {avoid_class}")
        return image, 0
    
    input_shape = image.shape
    r_tot = np.zeros(input_shape)  # Накопленное возмущение
    
    loop_i = 0
    pert_image = image.clone()
    
    while loop_i < max_iter:
        # Включаем градиенты для текущего шага
        pert_image = pert_image.clone().detach().requires_grad_(True)
        output = model(pert_image)
        
        # Текущий предсказанный класс
        current_class = torch.argmax(output).item()
        
        # ✅ Успех: ушли от целевого класса
        if current_class != avoid_class:
            break
        
        # Softmax для работы с вероятностями
        probs = F.softmax(output, dim=1)
        
        # Топ-(num_classes+1) наиболее вероятных классов
        top_k = min(num_classes + 1, output.shape[1])
        top_probs, top_indices = torch.topk(probs[0], k=top_k)
        
        # 🔹 Градиент для avoid_class (базовый)
        output[0, avoid_class].backward(retain_graph=True)
        grad_orig = pert_image.grad.detach().cpu().numpy().copy()
        pert_image.grad.zero_()
        
        pert = np.inf
        w_best = None
        
        # 🔍 Ищем ближайшую границу решения среди других классов
        for idx in top_indices:
            k = idx.item()
            if k == avoid_class:
                continue
            
            # Градиент для класса k
            output[0, k].backward(retain_graph=True)
            grad_k = pert_image.grad.detach().cpu().numpy().copy()
            pert_image.grad.zero_()
            
            # Разность градиентов и логитов
            w_k = grad_k - grad_orig
            f_k = (output[0, k] - output[0, avoid_class]).detach().cpu().numpy()
            
            # 📏 Расстояние до границы: |f| / ||w||
            pert_k = np.abs(f_k) / (np.linalg.norm(w_k.flatten()) + 1e-8)
            
            if pert_k < pert:
                pert = pert_k
                w_best = w_k
        
        if w_best is None:
            break
            
        # Вычисляем направление минимального возмущения
        norm_w = np.linalg.norm(w_best) + 1e-8
        r_i = (pert + 1e-4) * w_best / norm_w  # +1e-4 для численной стабильности
        r_tot = r_tot + r_i
        
        # Применяем возмущение с overshoot
        pert_image = image + torch.from_numpy((1 + overshoot) * r_tot).float().to(image.device)
        pert_image = torch.clamp(pert_image, -3, 3)  # Ограничение значений
        
        loop_i += 1
    
    # Финальное возмущение с overshoot
    r_tot = (1 + overshoot) * r_tot
    adv_image = image + torch.from_numpy(r_tot).float().to(image.device)
    adv_image = torch.clamp(adv_image, -3, 3)
    
    print(f"🔄 DeepFool завершён: {loop_i}/{max_iter} итераций")
    return adv_image, loop_i


def avoid_class_deepfool(model, image_path, avoid_class, 
                         num_classes=10, overshoot=0.02, max_iter=50):
    """
    Полный пайплайн атаки:
    1. Загрузка изображения в оригинальном разрешении
    2. Ресайз до 224×224 для модели + нормализация
    3. Применение DeepFool на 224×224
    4. Интерполяция возмущения обратно к оригинальному размеру
    5. Возврат нормализованного adversarial изображения
    """
    # 📷 1. Загружаем оригинал в полном разрешении
    original_np, orig_size = load_original_image(image_path)
    h, w = orig_size[1], orig_size[0]
    print(f"📷 Исходный размер изображения: {w}x{h}")

    # 🔄 2. Препроцессинг для модели (224×224)
    image_224, _ = load_and_preprocess_image(image_path, target_size=(224, 224))
    
    # 📊 Исходная вероятность целевого класса
    with torch.no_grad():
        init_output = model(image_224)
        init_prob = F.softmax(init_output, dim=1)[0, avoid_class].item()
    print(f"📉 Исходная вероятность класса {avoid_class}: {init_prob:.4f}")

    # ⚔️ 3. Применяем DeepFool атаку
    adv_224, n_iter = deepfool_avoid_class(
        model, image_224, avoid_class, 
        num_classes=num_classes, 
        overshoot=overshoot, 
        max_iter=max_iter
    )
    
    # 🎯 4. ВОЗВРАЩАЕМ ВОЗМУЩЕНИЕ К ОРИГИНАЛЬНОМУ РАЗМЕРУ
    # Вычисляем только шум (возмущение) на 224×224
    delta_224 = adv_224 - image_224
    
    # Растягиваем шум обратно к оригиналу (bilinear для плавности)
    delta_full = F.interpolate(delta_224, size=(h, w), 
                               mode='bilinear', align_corners=False)
    
    # Нормализуем оригинал в полном разрешении
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), 
                           std=IMAGENET_STD.tolist())
    ])
    original_norm = transform_norm(Image.fromarray(original_np)).unsqueeze(0)
    
    # Накладываем возмущение на оригинал
    adv_full = original_norm + delta_full
    adv_full = torch.clamp(adv_full, -3, 3)  # Ограничение диапазона
    
    print(f"✅ Атака завершена. Итоговое изображение: {w}x{h}")
    return adv_full

if __name__ == "__main__":
    IMAGE_PATH = '../data/test.png'
    
    # 🚀 Запуск атаки
    adversarial_img = avoid_class_deepfool(
        model, 
        IMAGE_PATH, 
        PHONE_CLASS,
        num_classes=10,      # Сколько соседних классов учитывать
        overshoot=0.02,      # Стандартное значение из статьи
        max_iter=50          # Максимум итераций
    )
    
    # 💾 Сохранение результата
    save_adversarial_image(adversarial_img, '../data/adversarial_no_phone_deepfool.png')
    
    # 🔍 Проверка: классификация после атаки
    with torch.no_grad():
        adv_224, _ = load_and_preprocess_image(
            '../data/adversarial_no_phone_deepfool.png', 
            target_size=(224, 224)
        )
        output = model(adv_224)
        probs = F.softmax(output, dim=1)
        
        pred_class = torch.argmax(probs).item()
        pred_prob = probs[0, pred_class].item()
        phone_prob = probs[0, PHONE_CLASS].item()

    
    detect(input_path='test.png')
    detect(input_path='adversarial_no_phone_deepfool.png')