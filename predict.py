import tensorflow as tf
import numpy as np
import cv2

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path, img_size):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    # No manual normalization — Rescaling layer inside model handles it
    img = np.expand_dims(img, axis=0)

    return img


def predict(image_path, model_path, img_size):

    model = load_trained_model(model_path)
    processed_img = preprocess_image(image_path, img_size)

    preds = model.predict(processed_img)

    class_index = np.argmax(preds)
    confidence = float(preds[0][class_index])

    print(f"Prediction : {CLASS_NAMES[class_index]}")
    print(f"Confidence : {confidence:.4f} ({confidence*100:.2f}%)")

    return CLASS_NAMES[class_index], confidence