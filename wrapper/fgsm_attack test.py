import tensorflow as tf
from keras.applications import ConvNeXtBase
from keras.applications.convnext import preprocess_input
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os
from model_functions import detect

# 1. Загружаем модель с гибким размером входа
base_model = ConvNeXtBase(weights='imagenet', include_top=False, input_shape=(None, None, 3))
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
predictions = tf.keras.layers.Dense(1000, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

PHONE_CLASS = 737 

def save_adversarial_image(img_tensor, output_path):
    # Безопасно извлекаем numpy-массив из TF тензора
    if isinstance(img_tensor, tf.Tensor):
        img = img_tensor.numpy()[0].copy()
    else:
        img = img_tensor[0].copy()
        
    # Обратная нормализация ConvNeXt: [-1, 1] -> [0, 255]
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)
    print(f"✅ Сохранено: {os.path.abspath(output_path)}")

def load_original_image(image_path):
    img = image.load_img(image_path)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def avoid_class_fgsm(model, image_path, avoid_class, epsilon=0.005, max_iter=20, max_eps=0.08):
    # 🔑 Преобразуем в TF тензор сразу, чтобы избежать проблем с типами в графе
    original_full = tf.constant(load_original_image(image_path), dtype=tf.float32)
    h, w = original_full.shape[1], original_full.shape[2]
    print(f"📷 Размер изображения: {w}x{h}")

    # Возмущение инициализируется нулями
    delta = tf.Variable(tf.zeros_like(original_full), dtype=tf.float32)
    target_label = tf.expand_dims(tf.one_hot(avoid_class, depth=1000), axis=0)

    init_prob = model(original_full)[0, avoid_class].numpy()
    print(f"📉 Исходная вероятность 'телефон': {init_prob:.4f}")

    for i in range(max_iter):
        # 🔑 КРИТИЧНО: Все операции, от которых зависит loss, должны быть ВНУТРИ tape
        with tf.GradientTape() as tape:
            tape.watch(delta)
            adv_full = original_full + delta
            predictions = model(adv_full)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(target_label, predictions))

        grads = tape.gradient(loss, delta)
        
        if grads is None:
            print("⚠️ Градиенты равны None. Проверьте, что вычисления внутри GradientTape.")
            break
            
        delta.assign_add(epsilon * tf.sign(grads))
        delta.assign(tf.clip_by_value(delta, -max_eps, max_eps))
        
        if i % 5 == 0:
            current_prob = model(original_full + delta)[0, avoid_class].numpy()
            print(f"   Итерация {i+1}: вероятность = {current_prob:.4f}")

    print(f"✅ Атака завершена. Максимальное изменение пикселя: {max_eps*127.5:.1f} из 255")
    return original_full + delta


IMAGE_PATH = '../data/test.png'
adversarial_img = avoid_class_fgsm(
    model, IMAGE_PATH, PHONE_CLASS, 
    epsilon=0.005,
    max_iter=20,
    max_eps=0.08
)
save_adversarial_image(adversarial_img, '../data/adversarial_no_phone.png')

print("\n🔍 Проверка детекции:")
detect(input_path=IMAGE_PATH)
detect(input_path='../data/adversarial_no_phone.png')