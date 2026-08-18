import tensorflow as tf
import os
from src.preprocessing import get_augmentation

AUTOTUNE = tf.data.AUTOTUNE

def load_data(data_path, img_size, batch_size):

    augmentation = get_augmentation()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_path, "train"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    ).map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_path, "val"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    ).prefetch(AUTOTUNE)

    return train_ds, val_ds