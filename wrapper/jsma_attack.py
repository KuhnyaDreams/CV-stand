import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import os

# ================== КОНФИГУРАЦИЯ ==================
MODEL_DIR = "../core/yolo26n_saved_model"
IMAGE_PATH = "../data/test.png"
OUTPUT_PATH = "../data/adversarial_no_phone_jsma.png"
NOISE_OUTPUT_PATH = "../data/jsma_noise.png"

AVOID_CLASS = 67  # "cell phone" в COCO
MAX_ITER = 10000
THETA = 0.1
GAMMA = 0.05
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
    
    # ✅ LANCZOS минимизирует потерю резкости при уменьшении
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

def save_noise_visualization(adv_tensor, original_tensor, meta, output_path):
    # ✅ Обработка tf.Variable и tf.Tensor
    if isinstance(adv_tensor, (tf.Tensor, tf.Variable)):
        adv_np = adv_tensor.numpy()
    else:
        adv_np = adv_tensor[0] if hasattr(adv_tensor, '__getitem__') else adv_tensor
    
    if isinstance(original_tensor, (tf.Tensor, tf.Variable)):
        orig_np = original_tensor.numpy()
    else:
        orig_np = original_tensor[0] if hasattr(original_tensor, '__getitem__') else original_tensor
    
    if adv_np.ndim == 4:
        adv_np = adv_np[0]
    if orig_np.ndim == 4:
        orig_np = orig_np[0]
    
    noise_padded = adv_np - orig_np
    noise_cropped = noise_padded[meta['top']:meta['top']+meta['new_h'], 
                                 meta['left']:meta['left']+meta['new_w'], :]
    
    # ✅ Нормализуем и конвертируем в uint8 ДО ресайза
    n_min, n_max = noise_cropped.min(), noise_cropped.max()
    if n_max > n_min:
        noise_norm = ((noise_cropped - n_min) / (n_max - n_min) * 255).astype(np.uint8)
    else:
        noise_norm = np.zeros_like(noise_cropped, dtype=np.uint8)
        
    noise_img = Image.fromarray(noise_norm)
    noise_resized = noise_img.resize((meta['orig_w'], meta['orig_h']), Image.Resampling.LANCZOS)
    noise_resized.save(output_path)
    print(f"👁️ Noise visualization saved: {os.path.abspath(output_path)}")

def extract_class_scores(predictions, class_id):
    confidences = predictions[0, :, 4]
    detected_classes = predictions[0, :, 5]
    mask = tf.abs(detected_classes - tf.cast(class_id, tf.float32)) < 0.1
    masked_conf = tf.where(mask, confidences, tf.zeros_like(confidences))
    return tf.reduce_max(masked_conf)

def jsma_avoid_class_yolo(model, image_path, avoid_class, 
                          max_iter=MAX_ITER, theta=THETA, gamma=GAMMA):
    original_tensor, meta = letterbox_preprocess(image_path)
    print(f"📷 Original: {meta['orig_w']}x{meta['orig_h']}, "
          f"Resized: {meta['new_w']}x{meta['new_h']}, Pad: T={meta['top']},L={meta['left']}")

    init_pred = model(original_tensor)
    init_score = extract_class_scores(init_pred, avoid_class)
    print(f"🎯 Initial max score for class {avoid_class}: {init_score.numpy():.4f}")

    adv = tf.Variable(original_tensor, dtype=tf.float32, trainable=False)
    modified_mask = tf.zeros_like(adv)
    total_elements = tf.size(original_tensor).numpy()

    content_mask = tf.ones((meta['new_h'], meta['new_w'], 3), dtype=tf.float32)
    valid_area_mask = tf.pad(
        content_mask,
        paddings=[[meta['top'], meta['bottom']], 
                  [meta['left'], meta['right']], 
                  [0, 0]],
        constant_values=0.0
    )
    valid_area_mask = tf.expand_dims(valid_area_mask, 0)

    for iteration in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            pred = model(adv)
            current_score = extract_class_scores(pred, avoid_class)
            loss = current_score

        grads = tape.gradient(loss, adv)
        if grads is None or tf.reduce_max(tf.abs(grads)) == 0:
            print("⚠️ Zero gradients")
            break

        saliency = tf.abs(grads) * tf.cast(modified_mask == 0, tf.float32) * valid_area_mask
            
        if tf.reduce_max(saliency) == 0:
            print("⚠️ No available pixels in content area")
            break

        flat_saliency = tf.reshape(saliency, [-1])
        num_mod = min(3, tf.math.count_nonzero(flat_saliency > 0).numpy())
        if num_mod == 0:
            break

        top_indices = tf.math.top_k(flat_saliency, k=num_mod).indices
        
        # ✅ Оптимизация: используем NumPy для эффективного обновления вместо tensor_scatter_nd_update
        adv_np = adv.numpy()
        mask_np = modified_mask.numpy()
        grads_np = grads.numpy()
        
        for idx in top_indices.numpy():
            idx_int = int(idx)
            if mask_np.flat[idx_int] == 0:
                direction = -np.sign(grads_np.flat[idx_int])
                adv_np.flat[idx_int] = np.clip(adv_np.flat[idx_int] + gamma * direction, 0.0, 1.0)
                mask_np.flat[idx_int] = 1
        
        # Обновляем TensorFlow переменные
        adv.assign(tf.constant(adv_np, dtype=tf.float32))
        modified_mask = tf.constant(mask_np, dtype=tf.float32)

        if iteration % 10 == 0 or iteration == max_iter - 1:
            new_pred = model(adv)
            new_score = extract_class_scores(new_pred, avoid_class)
            pct = 100.0 * tf.reduce_sum(modified_mask).numpy() / total_elements
            print(f"🔁 Iter {iteration+1}: score={new_score.numpy():.4f}, modified={pct:.2f}%")

        if current_score < 0.05:
            print(f"✅ Attack succeeded at iteration {iteration+1}")
            break
        if tf.reduce_sum(modified_mask) >= int(theta * total_elements):
            print(f"⛔ L0 limit reached")
            break

    return adv, original_tensor, meta

def save_adversarial_image(img_tensor, meta, output_path):
    restored_np = unpad_and_restore(img_tensor, meta)
    Image.fromarray(restored_np).save(output_path)
    print(f"✅ Adversarial saved: {os.path.abspath(output_path)} ({meta['orig_w']}x{meta['orig_h']})")

# ================== ЗАПУСК ==================
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️ GPU: {[g.name for g in gpus]}")

    print(f"📦 Loading model: {MODEL_DIR}")
    yolo_model = load_yolo_keras(MODEL_DIR)
    print(f"✅ Model: input={yolo_model.input_shape}, output={yolo_model.output_shape}")

    print(f"🚀 Starting JSMA attack (avoid class {AVOID_CLASS})...")
    adversarial, original_tensor, meta = jsma_avoid_class_yolo(
        model=yolo_model,
        image_path=IMAGE_PATH,
        avoid_class=AVOID_CLASS,
        max_iter=MAX_ITER,
        theta=THETA,
        gamma=GAMMA
    )
    
    save_adversarial_image(adversarial, meta, OUTPUT_PATH)
    save_noise_visualization(adversarial, original_tensor, meta, NOISE_OUTPUT_PATH)
    
    print("\n💡 Verify: from ultralytics import YOLO; YOLO('...').predict('adversarial_no_phone.png')")
    
    
    from model_functions import detect
    detect(IMAGE_PATH)
    detect(OUTPUT_PATH)