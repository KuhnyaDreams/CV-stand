import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os
from model_functions import detect
# 1. Загружаем модель
model = ResNet50(weights='imagenet')

# Класс "cellular telephone" в ImageNet
PHONE_CLASS = 737 

def save_adversarial_image(img_tensor, output_path):
    img = img_tensor[0].copy()
    img += [103.939, 116.779, 123.68]  # Возвращаем вычтенное среднее
    img = img[..., ::-1]                # BGR -> RGB
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)
    print(f"✅ Сохранено: {os.path.abspath(output_path)}")

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # preprocess_input переводит в BGR и вычитает среднее ImageNet
    return preprocess_input(img_array)

def load_original_image(image_path):
    """Загружает изображение в ИСХОДНОМ размере (без target_size)"""
    img = image.load_img(image_path)  # <-- Убран target_size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def avoid_class_fgsm(model, image_path, avoid_class, epsilon=1.0, max_iter=15):
    # 1. Оригинал в полном разрешении
    original_full = load_original_image(image_path)
    h, w = original_full.shape[1], original_full.shape[2]
    print(f"📷 Исходный размер изображения: {w}x{h}")

    # 2. Сжимаем ТОЛЬКО для модели (ResNet требует 224x224)
    original_224 = tf.image.resize(original_full[0], (224, 224))
    adv_224 = tf.Variable(tf.expand_dims(original_224, axis=0))

    target_label = tf.expand_dims(tf.one_hot(avoid_class, depth=1000), axis=0)

    init_prob = model(adv_224)[0, avoid_class].numpy()
    print(f"📉 Исходная вероятность 'телефон' (на 224x224): {init_prob:.4f}")

    # 3. Цикл атаки
    for i in range(max_iter):
        with tf.GradientTape() as tape:
            predictions = model(adv_224)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(target_label, predictions))

        grads = tape.gradient(loss, adv_224)
        adv_224.assign_add(epsilon * tf.sign(grads))
        adv_224.assign(tf.clip_by_value(adv_224, -125.0, 155.0))

        if i % 3 == 0:
            current_prob = model(adv_224)[0, avoid_class].numpy()
            print(f"   Итерация {i+1}: вероятность = {current_prob:.4f}")

    # 4. 🔑 ВОЗВРАЩАЕМ РЕЗУЛЬТАТ К ИСХОДНОМУ РАЗМЕРУ
    # Вычисляем только возмущение (шум)
    delta_224 = adv_224 - tf.expand_dims(original_224, axis=0)
    
    # Растягиваем шум обратно до оригинала (bilinear сохраняет плавность)
    delta_full = tf.image.resize(delta_224[0], (h, w), method='bilinear')
    delta_full = tf.expand_dims(delta_full, axis=0)

    # Накладываем шум на оригинал
    adv_full = original_full + delta_full.numpy()
    adv_full = tf.clip_by_value(adv_full, -125.0, 155.0).numpy()

    print(f"✅ Атака завершена. Итоговое изображение: {w}x{h}")
    return adv_full


IMAGE_PATH = '../data/test.png'
adversarial_img = avoid_class_fgsm(model, IMAGE_PATH, PHONE_CLASS, epsilon=1.0, max_iter=15)
save_adversarial_image(adversarial_img, '../data/adversarial_no_phone.png')
detect(input_path='test.png')
detect(input_path='adversarial_no_phone.png')