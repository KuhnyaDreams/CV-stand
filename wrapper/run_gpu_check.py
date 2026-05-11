import tensorflow as tf
print("TF версия:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(f"Найдено GPU: {len(gpus)}")
for gpu in gpus:
    print(f" - {gpu}")