import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import os
from api.model_functions import detect

# ================== КОНФИГУРАЦИЯ ==================
MODEL_DIR = "../core/yolo26n_saved_model"
IMAGE_PATH = "../data/test.png"
OUTPUT_PATH = "../data/adversarial_no_phone_pgd.png"
AVOID_CLASS = 67  # "cell phone" в COCO
EPSILON = 20 / 255.0  # Радиус возмущения (в диапазоне 0-1) - УВЕЛИЧЕН
ALPHA = 4 / 255.0    # Размер шага итерации - УВЕЛИЧЕН
MAX_ITER = 30  # УВЕЛИЧЕН для лучшей сходимости
IMG_SIZE = 640
# ===================================================

def load_yolo_keras(model_dir, img_size=IMG_SIZE):
    layer = keras.layers.TFSMLayer(model_dir, call_endpoint='serving_default')
    inputs = keras.Input(shape=(img_size, img_size, 3), dtype=tf.float32)
    output = layer(inputs)
    if isinstance(output, dict):
        output = list(output.values())[0]
    return keras.Model(inputs=inputs, outputs=output)

def letterbox_preprocess(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    r = min(target_size[0] / orig_w, target_size[1] / orig_h)
    new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
    
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    pad_w = (target_size[0] - new_w) / 2
    pad_h = (target_size[1] - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    
    padded = np.pad(img_array, ((top, bottom), (left, right), (0, 0)), 
                    mode='constant', constant_values=0.447)
    padded = np.expand_dims(padded, axis=0)
    
    meta = {
        'orig_w': orig_w, 'orig_h': orig_h,
        'new_w': new_w, 'new_h': new_h,
        'top': top, 'bottom': bottom, 'left': left, 'right': right
    }
    return tf.constant(padded, dtype=tf.float32), meta

def unpad_and_restore(adv_tensor, meta):
    # ✅ Обработка tf.Variable и tf.Tensor
    if isinstance(adv_tensor, (tf.Tensor, tf.Variable)):
        adv_np = adv_tensor.numpy()
    else:
        adv_np = adv_tensor
        
    if adv_np.ndim == 4:
        adv_np = adv_np[0]
        
    crop = adv_np[meta['top']:meta['top']+meta['new_h'], 
                  meta['left']:meta['left']+meta['new_w'], :]
    
    # ✅ Конвертируем в uint8 ДО ресайза, чтобы PIL применил качественный LANCZOS
    img_crop = Image.fromarray((crop * 255).clip(0, 255).astype(np.uint8))
    img_orig = img_crop.resize((meta['orig_w'], meta['orig_h']), Image.Resampling.LANCZOS)
    return np.array(img_orig, dtype=np.uint8)

def extract_class_score(predictions, class_id):
    """Извлекает максимальный confidence для целевого класса"""
    confidences = predictions[0, :, 4]
    detected_classes = predictions[0, :, 5]
    mask = tf.abs(detected_classes - tf.cast(class_id, tf.float32)) < 0.1
    masked_conf = tf.where(mask, confidences, -1.0 * tf.ones_like(confidences))
    return tf.reduce_max(masked_conf + 1.0)

def create_pgd_attack_loss(predictions, avoid_class):
    """Улучшенная loss функция для PGD атаки."""
    confidences = predictions[0, :, 4]
    detected_classes = predictions[0, :, 5]
    
    avoid_mask = tf.abs(detected_classes - tf.cast(avoid_class, tf.float32)) < 0.1
    target_confs = tf.where(avoid_mask, confidences, tf.zeros_like(confidences))
    target_score = tf.reduce_max(target_confs)
    
    other_confs = tf.where(~avoid_mask, confidences, tf.zeros_like(confidences))
    other_score = tf.reduce_mean(other_confs)
    
    # Более агрессивная loss для PGD
    loss = target_score - 0.2 * other_score + 1e-6
    return loss

def pgd_avoid_class_yolo(model, image_tensor, avoid_class, epsilon=EPSILON, 
                         alpha=ALPHA, max_iter=MAX_ITER):
    """
    PGD (Projected Gradient Descent) атака для YOLO.
    
    Алгоритм:
    1. Инициализируем adv = original + random noise (в пределах epsilon)
    2. На каждой итерации:
       - Вычисляем градиент loss w.r.t. adv
       - Делаем шаг в направлении возрастания loss
       - Проецируем обратно в epsilon-ball вокруг original
    """
    original_tensor = image_tensor
    adv = tf.Variable(original_tensor, dtype=tf.float32)
    
    # Проверка начального состояния
    with tf.GradientTape() as tape:
        tape.watch(adv)
        pred = model(adv)
        init_score = extract_class_score(pred, avoid_class)
    
    print(f"🎯 Initial score for class {avoid_class}: {init_score.numpy():.4f}")
    
    if init_score < 0.05:
        print(f"⚠️ Target class already has low confidence")
        return adv, 0
    
    for iteration in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            pred = model(adv)
            # ✅ Используем улучшенную loss функцию
            loss = create_pgd_attack_loss(pred, avoid_class)
        
        # Вычисляем градиент
        grads = tape.gradient(loss, adv)
        max_grad = tf.reduce_max(tf.abs(grads)) if grads is not None else 0.0
        if grads is None or max_grad < 1e-8:
            if iteration > 10:
                print("⚠️ Zero gradients, stopping")
                break
            grads = tf.ones_like(adv) * 1e-5
        
        # 🔹 PGD шаг: движемся вдоль градиента
        # Чтобы уменьшить confidence, движемся в направлении -sign(grad)
        perturbation = alpha * (-tf.sign(grads))
        adv_new = adv + perturbation
        
        # 🔹 Проекция: ограничиваем возмущение в Linf ball вокруг original
        # Используем L∞ норму вместо L2 для лучшей efficacy
        delta = tf.clip_by_value(adv_new - original_tensor, -epsilon, epsilon)
        adv_new = original_tensor + delta
        
        # Ограничиваем пиксели в диапазоне [0, 1]
        adv_new = tf.clip_by_value(adv_new, 0.0, 1.0)
        adv.assign(adv_new)
        
        # ✅ Вычисляем current_score на каждой итерации
        with tf.GradientTape() as tape:
            tape.watch(adv)
            pred = model(adv)
            current_score = extract_class_score(pred, avoid_class)
        
        # Логирование
        if (iteration + 1) % 3 == 0 or iteration == max_iter - 1:
            print(f"🔁 Iter {iteration+1}/{max_iter}: score={current_score.numpy():.4f}")
        
        # ✅ Успех: ушли от целевого класса
        if current_score < 0.05:
            print(f"✅ Attack succeeded at iteration {iteration+1}")
            break
    
    print(f"🔄 PGD завершён: {iteration+1}/{max_iter} итераций")
    return adv, iteration+1

def save_adversarial_image(img_tensor, meta, output_path):
    """Сохраняет adversarial изображение в оригинальном размере"""
    restored_np = unpad_and_restore(img_tensor, meta)
    Image.fromarray(restored_np).save(output_path)
    print(f"✅ Adversarial saved: {os.path.abspath(output_path)}")

# ================== ЗАПУСК ==================
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️ GPU: {[g.name for g in gpus]}")

    print(f"📦 Loading YOLO model: {MODEL_DIR}")
    yolo_model = load_yolo_keras(MODEL_DIR)
    print(f"✅ Model: input={yolo_model.input_shape}, output={yolo_model.output_shape}")

    print(f"🚀 Starting PGD attack (avoid class {AVOID_CLASS})...")
    original_tensor, meta = letterbox_preprocess(IMAGE_PATH)
    print(f"📷 Original: {meta['orig_w']}x{meta['orig_h']}, "
          f"Resized: {meta['new_w']}x{meta['new_h']}, Pad: T={meta['top']},L={meta['left']}")
    
    adversarial, n_iter = pgd_avoid_class_yolo(
        model=yolo_model,
        image_tensor=original_tensor,
        avoid_class=AVOID_CLASS,
        epsilon=EPSILON,
        alpha=ALPHA,
        max_iter=MAX_ITER
    )
    
    save_adversarial_image(adversarial, meta, OUTPUT_PATH)
    
    print("\n🔍 Проверка детекции:")
    detect(IMAGE_PATH)
    detect(OUTPUT_PATH)
