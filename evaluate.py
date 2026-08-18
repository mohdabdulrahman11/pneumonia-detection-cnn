from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, val_ds, class_names):

    # --- Predict on entire validation set at once
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # --- Extract true labels from dataset
    y_true = np.concatenate(
        [np.argmax(labels.numpy(), axis=1) for _, labels in val_ds]
    )

    # --- Metrics
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('artifacts/confusion_matrix.png')
    plt.show()
    print("\nConfusion matrix saved to artifacts/confusion_matrix.png")