# Pneumonia Detection using CNN & Transfer Learning



## Problem
Pneumonia is a life-threatening condition where early detection from chest
X-ray images can significantly improve patient outcomes. This project builds
a deep learning model to classify chest X-rays as **Normal** or **Pneumonia**.

## Dataset
- **Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** NORMAL vs PNEUMONIA (binary classification)
- **Splits:** Train / Validation / Test

## Project Structure
```
pneumonia-detection-cnn/
│
├── src/
│   ├── data_loader.py      # TF dataset pipeline + augmentation
│   ├── preprocessing.py    # Data augmentation layers
│   ├── model.py            # ResNet50 transfer learning architecture
│   ├── train.py            # Compile + train with callbacks
│   ├── evaluate.py         # Metrics + confusion matrix
│   ├── predict.py          # Single image inference
│   └── utils.py            # Config loader
├── config/
│   └── config.yaml         # All hyperparameters
├── artifacts/              # Saved model output
├── main.py                 # Entry point
└── requirements.txt
```

## Model Architecture (ResNet50 Transfer Learning)
```
Input (224x224x3)
    ↓
ResNet50 preprocess_input     # Normalize pixels for ResNet50
    ↓
ResNet50 (pretrained ImageNet, frozen)   # Feature extraction
    ↓
GlobalAveragePooling2D        # Compact feature vector
    ↓
BatchNormalization            # Normalize feature values (μ=0, σ=1)
    ↓
Dense(256, relu)              # Fully connected layer
    ↓
Dropout(0.5)                  # Prevent overfitting
    ↓
Dense(2, softmax)             # Output: NORMAL / PNEUMONIA probability
```

## Pipeline
1. **Data Loading** — `tf.keras.utils.image_dataset_from_directory` with
   stratified train/val/test splits
2. **Data Augmentation** — RandomFlip, RandomRotation, RandomZoom applied
   to training set only; handles limited healthcare imaging data
3. **Preprocessing** — `resnet50.preprocess_input` for ResNet-compatible
   pixel normalization
4. **Training** — Adam optimizer (lr=0.0001), categorical crossentropy,
   EarlyStopping (patience=3) + ModelCheckpoint
5. **Evaluation** — Accuracy, precision, recall, AUC + confusion matrix
6. **Inference** — Single image prediction via `predict.py`

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Image Size | 224 x 224 |
| Batch Size | 32 |
| Epochs | 10 (early stopping) |
| Learning Rate | 0.0001 |
| Dropout | 0.5 |
| Optimizer | Adam |

## Tech Stack
Python, TensorFlow, Keras, ResNet50, Transfer Learning,
scikit-learn, OpenCV, Matplotlib, Seaborn, Google Colab
