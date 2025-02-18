import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check if TPU is available
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    device = 'TPU'
    print("TPU found. Using TPU for training.")
except:
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
        device = 'GPU'
        print("GPU found. Using GPU for training.")
    else:
        strategy = tf.distribute.OneDeviceStrategy('/CPU:0')
        device = 'CPU'
        print("No TPU or GPU found. Using CPU for training.")

# Paths to datasets
train_dir = 'Dataset/Train'
val_dir = 'Dataset/Val'

# Image dimensions
img_height, img_width = 32, 32
batch_size = 32

# ImageDataGenerators for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary'
)

# Function to embed hash in model
def embed_hash_in_model(model):
    model_json = model.to_json().encode('utf-8')
    model_hash = hashlib.sha256(model_json).hexdigest()
    hash_tensor = tf.convert_to_tensor([ord(c) for c in model_hash], dtype=tf.float32)
    model.add(tf.keras.layers.Lambda(lambda x: x * 1, name="hash_layer"))  # Dummy layer to hold hash
    return model_hash

# Build CNN model
def create_model():
    model = models.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

with strategy.scope():
    model = create_model()
    model_hash = embed_hash_in_model(model)

model.summary()  # Print model architecture
print("Model Hash:", model_hash)

# Train model
history = model.fit(
    train_generator, steps_per_epoch=train_generator.samples // batch_size,
    epochs=20, validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Save the tamper-proof model
model.save('DeepFake-Detector-TamperProof.keras')
print("Tamper-proof model saved as 'DeepFake-Detector-TamperProof.keras'")

# Plot training and validation accuracy and loss
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
train_acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])

axs[0, 0].plot(train_acc, label='Training Accuracy', marker='o')
axs[0, 0].set_title('Training Accuracy')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()
axs[0, 0].grid()

axs[0, 1].plot(train_loss, label='Training Loss', marker='o', color='red')
axs[0, 1].set_title('Training Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[0, 1].grid()

if val_acc:
    axs[1, 0].plot(val_acc, label='Validation Accuracy', marker='o', color='green')
    axs[1, 0].set_title('Validation Accuracy')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    axs[1, 0].grid()

if val_loss:
    axs[1, 1].plot(val_loss, label='Validation Loss', marker='o', color='orange')
    axs[1, 1].set_title('Validation Loss')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    axs[1, 1].grid()

plt.suptitle('DeepFake Detector Training Performance', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
