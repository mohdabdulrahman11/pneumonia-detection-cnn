from src.utils import load_config
from src.data_loader import load_data
from src.model import build_model
from src.train import compile_model, train_model
from src.evaluate import evaluate_model

config = load_config("config/config.yaml")

train_ds, val_ds = load_data(
    config["DATA_PATH"],
    config["IMAGE_SIZE"],
    config["BATCH_SIZE"]
)

class_names = train_ds.class_names
num_classes = len(class_names)

model = build_model(num_classes, config["IMAGE_SIZE"])
model = compile_model(model, config["LEARNING_RATE"])

history = train_model(
    model,
    train_ds,
    val_ds,
    config["EPOCHS"],
    config["MODEL_SAVE_PATH"]   # ← fix: pass save path from config
)

evaluate_model(model, val_ds, class_names)