import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict_vault_image(img_path, model, class_names):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    print(f"Prediction: {class_names[predicted_idx]} ({confidence * 100:.2f}% confidence)")

# Example execution in testing2.py
if __name__ == "__main__":
    class_names = ["label_a", "label_b", "label_c"] # Match your vault structure
    model = tf.keras.models.load_model("saved_vault_model.h5")
    predict_vault_image("path/to/vault/test/label_a/sample.jpg", model, class_names)