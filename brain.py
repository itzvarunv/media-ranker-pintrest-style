import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_vault_model(num_classes, img_shape=(224, 224, 3)):
    # Load pretrained backbone
    base_model = MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze pretrained weights
    
    # Build custom top layers for your specific labels
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax') # Text label classes
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model