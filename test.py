import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from brain import build_vault_model

# Setup test dataset from vault
test_ds = image_dataset_from_directory(
    "path/to/vault/test",
    image_size=(224, 224),
    batch_size=32
)

# Initialize and load weights
model = build_vault_model(num_classes=len(test_ds.class_names))
model.load_weights("saved_vault_model.h5")

# Evaluate performance
loss, acc = model.evaluate(test_ds)
print(f"Vault Test Accuracy: {acc * 100:.2f}%")