import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import os
from model_functions import detect

# ================== КОНФИГУРАЦИЯ ==================
MODEL_DIR = "../core/yolo26n_saved_model"
IMAGE_PATH = "../data/test.png"
OUTPUT_PATH = "../data/adversarial_no_phone_fgsm.png"
AVOID_CLASS = 67  # "cell phone" в COCO
EPSILON = 0.02   # Шаг возмущения (в диапазоне 0.0-1.0) - УВЕЛИЧЕН
MAX_ITER = 150     # Количество шагов I-FGSM - УВЕЛИЧЕН
IMG_SIZE = 640
# ===================================================

def load_yolo_keras(model_dir, img_size=IMG_SIZE):
    layer = keras.layers.TFSMLayer(model_dir, call_endpoint='serving_default')
    inputs = keras.Input(shape=(img_size, img_size, 3), dtype=tf.float32)
    output = layer(inputs)
    if isinstance(output, dict):
        output = list(output.values())[0]
    model = keras.Model(inputs=inputs, outputs=output)
    model.trainable = False
    return model

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

def extract_yolo_class_score(predictions, class_id):
    confidences = predictions[0, :, 4]
    detected_classes = predictions[0, :, 5]
    mask = tf.abs(detected_classes - tf.cast(class_id, tf.float32)) < 0.1
    masked_conf = tf.where(mask, confidences, -1.0 * tf.ones_like(confidences))
    return tf.reduce_max(masked_conf + 1.0)

def create_fgsm_attack_loss(predictions, avoid_class):
    """Улучшенная loss функция для FGSM атаки."""
    confidences = predictions[0, :, 4]
    detected_classes = predictions[0, :, 5]
    
    avoid_mask = tf.abs(detected_classes - tf.cast(avoid_class, tf.float32)) < 0.1
    target_confs = tf.where(avoid_mask, confidences, tf.zeros_like(confidences))
    target_score = tf.reduce_max(target_confs)
    
    other_confs = tf.where(~avoid_mask, confidences, tf.zeros_like(confidences))
    other_score = tf.reduce_mean(other_confs)
    
    loss = target_score - 0.15 * other_score + 1e-6
    return loss

def ifgsm_avoid_yolo(model, image_path, avoid_class, epsilon=EPSILON, max_iter=MAX_ITER):
    original_tensor, meta = letterbox_preprocess(image_path)
    print(f"📷 Original: {meta['orig_w']}x{meta['orig_h']}, "
          f"Resized: {meta['new_w']}x{meta['new_h']}, Pad: T={meta['top']},L={meta['left']}")

    init_score = float(extract_yolo_class_score(model(original_tensor), avoid_class).numpy())
    print(f"🎯 Initial score for class {avoid_class}: {init_score:.4f}")

    adv = tf.Variable(original_tensor, dtype=tf.float32)

    for i in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            pred = model(adv)
            loss = create_fgsm_attack_loss(pred, avoid_class)  # ✅ Улучшенная loss

        grads = tape.gradient(loss, adv)
        max_grad = float(tf.reduce_max(tf.abs(grads))) if grads is not None else 0.0
        if grads is None or max_grad < 1e-8:
            if i > 20:
                print("⚠️ Zero gradients, stopping.")
                break
            grads = tf.ones_like(adv) * 1e-5

        perturbation = epsilon * tf.sign(grads + 1e-7)
        adv.assign(adv - perturbation)
        adv.assign(tf.clip_by_value(adv, 0.0, 1.0))

        # ✅ 3. Вычисляем score НА КАЖДОЙ итерации и приводим к float
        current_score = float(extract_yolo_class_score(model(adv), avoid_class).numpy())

        if (i + 1) % 3 == 0 or i == max_iter - 1:
            print(f"   Iter {i+1}/{max_iter}: score = {current_score:.4f}")

        if current_score < 0.05:
            print(f"✅ Attack succeeded early at iter {i+1}")
            break

    return adv, original_tensor, meta

def save_adversarial_image(img_tensor, meta, output_path):
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

    print(f"🚀 Starting I-FGSM attack on YOLO (avoid class {AVOID_CLASS})...")
    adversarial, original, meta = ifgsm_avoid_yolo(
        model=yolo_model,
        image_path=IMAGE_PATH,
        avoid_class=AVOID_CLASS,
        epsilon=EPSILON,
        max_iter=MAX_ITER
    )
    
    save_adversarial_image(adversarial, meta, OUTPUT_PATH)
    
    print("\n🔍 Проверка детекции:")
    detect(IMAGE_PATH)
    detect(OUTPUT_PATH)