import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import os
from api.model_functions import detect

# ================== КОНФИГУРАЦИЯ ==================
MODEL_DIR = "../core/yolo26n_saved_model"
IMAGE_PATH = "../data/test.png"
OUTPUT_PATH = "../data/adversarial_no_phone_deepfool.png"
AVOID_CLASS = 67  # "cell phone" в COCO
NUM_CLASSES = 10  # Количество соседних классов для поиска границы
OVERSHOOT = 0.05  # Увеличен для более агрессивной атаки
MAX_ITER = 100    # Увеличен для лучшей сходимости
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
    """Извлекает максимальный confidence для целевого класса из YOLO predictions"""
    confidences = predictions[0, :, 4]
    detected_classes = predictions[0, :, 5]
    mask = tf.abs(detected_classes - tf.cast(class_id, tf.float32)) < 0.1
    masked_conf = tf.where(mask, confidences, -1.0 * tf.ones_like(confidences))
    return tf.reduce_max(masked_conf + 1.0)  # Сдвиг для обработки пустого случая

def deepfool_avoid_yolo(model, image_tensor, avoid_class, num_classes=NUM_CLASSES,
                        overshoot=OVERSHOOT, max_iter=MAX_ITER):
    """
    DeepFool атака для YOLO модели.
    Ищет минимальное возмущение, чтобы уменьшить confidence целевого класса.
    """
    image = tf.Variable(image_tensor, dtype=tf.float32)
    
    # Проверка начального состояния
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image)
        init_score = extract_class_score(pred, avoid_class)
    
    print(f"🎯 Initial score for class {avoid_class}: {init_score.numpy():.4f}")
    
    if init_score < 0.05:
        print(f"⚠️ Target class already has low confidence")
        return image, 0
    
    r_tot = np.zeros(image_tensor.shape)  # Накопленное возмущение
    
    for iteration in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(image)
            pred = model(image)
            current_score = extract_class_score(pred, avoid_class)
        
        print(f"🔁 Iter {iteration+1}/{max_iter}: score={current_score.numpy():.4f}")
        
        # ✅ Успех: ушли от целевого класса
        if current_score < 0.05:
            print(f"✅ Attack succeeded at iteration {iteration+1}")
            break
        
        # Градиент для целевого класса
        grads = tape.gradient(current_score, image)
        if grads is None or tf.reduce_max(tf.abs(grads)) < 1e-7:
            print("⚠️ Zero gradients, stopping")
            break
        
        grad_avoid = grads.numpy().copy()
        
        # Получаем все confidence scores для поиска ближайшей границы
        pred_current = model(image)
        confidences = pred_current[0, :, 4]  # [N]
        detected_classes = pred_current[0, :, 5]  # [N]
        
        # Построим словарь: class_id -> max_confidence для каждого класса
        class_scores = {}
        for detection_idx in range(detected_classes.shape[0]):
            class_id = int(detected_classes[detection_idx].numpy())
            conf = float(confidences[detection_idx].numpy())
            
            if class_id not in class_scores:
                class_scores[class_id] = conf
            else:
                class_scores[class_id] = max(class_scores[class_id], conf)
        
        # Ищем ближайшую границу решения
        pert_min = np.inf
        w_best = None
        best_class = None
        
        current_score_val = float(current_score.numpy())
        
        for class_id in class_scores.keys():
            if class_id == avoid_class:
                continue
            
            # Вычисляем градиент для этого класса
            with tf.GradientTape() as tape_k:
                tape_k.watch(image)
                pred_k = model(image)
                score_k = extract_class_score(pred_k, class_id)
            
            grad_k = tape_k.gradient(score_k, image)
            if grad_k is None:
                continue
            
            grad_k_np = grad_k.numpy().copy()
            
            # Разность градиентов
            w_k = grad_k_np - grad_avoid
            norm_w = np.linalg.norm(w_k.flatten()) + 1e-8
            
            if norm_w < 1e-7:
                continue
            
            # Разность confidence scores
            score_k_val = float(score_k.numpy())
            f_k = score_k_val - current_score_val
            
            # 📏 Расстояние до границы: |f| / ||w||
            pert_k = np.abs(f_k) / norm_w
            
            if pert_k < pert_min:
                pert_min = pert_k
                w_best = w_k
                best_class = class_id
        
        if w_best is None or pert_min == np.inf:
            print("⚠️ No suitable boundary found, using gradient direction")
            # Используем только градиент целевого класса
            w_best = grad_avoid
            pert_min = 1e-2
        
        # Направление минимального возмущения
        norm_w = np.linalg.norm(w_best.flatten()) + 1e-8
        r_i = (pert_min + 1e-4) * w_best / norm_w
        r_tot = r_tot + r_i
        
        # Применяем возмущение с overshoot
        pert_image = image_tensor + tf.constant((1 + overshoot) * r_tot, dtype=tf.float32)
        pert_image = tf.clip_by_value(pert_image, 0.0, 1.0)
        
        image.assign(pert_image)
    
    # Финальное возмущение с overshoot
    r_tot = (1 + overshoot) * r_tot
    adv_image = image_tensor + tf.constant(r_tot, dtype=tf.float32)
    adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)
    
    print(f"🔄 DeepFool завершён: {iteration+1}/{max_iter} итераций")
    return adv_image, iteration+1

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

    print(f"🚀 Starting DeepFool attack (avoid class {AVOID_CLASS})...")
    original_tensor, meta = letterbox_preprocess(IMAGE_PATH)
    print(f"📷 Original: {meta['orig_w']}x{meta['orig_h']}, "
          f"Resized: {meta['new_w']}x{meta['new_h']}, Pad: T={meta['top']},L={meta['left']}")
    
    adversarial, n_iter = deepfool_avoid_yolo(
        model=yolo_model,
        image_tensor=original_tensor,
        avoid_class=AVOID_CLASS,
        num_classes=NUM_CLASSES,
        overshoot=OVERSHOOT,
        max_iter=MAX_ITER
    )
    
    save_adversarial_image(adversarial, meta, OUTPUT_PATH)
    
    print("\n🔍 Проверка детекции:")
    detect(IMAGE_PATH)
    detect(OUTPUT_PATH)
