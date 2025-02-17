import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check if GPU is available, otherwise fallback to CPU
if tf.config.list_physical_devices('GPU'):
    device = '/GPU:0'  # If GPU is available, use GPU
    print("GPU found. Using GPU for training.")
else:
    device = '/CPU:0'  # If GPU is not available, use CPU
    print("No GPU found. Using CPU for training.")

# Paths to the training and validation directories
train_dir = 'Dataset/Train'
val_dir = 'Dataset/Val'

# Image dimensions
img_height = 32  # Image size is 32x32
img_width = 32
batch_size = 32

# Create ImageDataGenerators for training and validation datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=30,  # Random rotations for augmentation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Random shear transformations
    zoom_range=0.2,  # Random zooms
    horizontal_flip=True,  # Random horizontal flips
    fill_mode='nearest'  # Fill missing pixels during transformation
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load the images from the directories and apply the transformations
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  # Since we have two classes (REAL, FAKE)
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the CNN model with explicit Input layer
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),  # Explicit input layer
    # First Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Second Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Third Convolutional Block
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Flatten and Fully Connected Layers
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),  # Dropout to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary classification (0 = REAL, 1 = FAKE)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and store the training history
with tf.device(device):  # Specify the device to be used (CPU or GPU)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=20,  # Increased epochs due to large dataset
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )

# Save the trained model
model.save('deepfake_detection_model_large_dataset_32x32_fixed_with_device.h5')
print("Model saved as 'deepfake_detection_model_large_dataset_32x32_fixed_with_device.h5'")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Test Accuracy: {test_accuracy}")

# Visualize the training and validation accuracy and loss
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Get history data
train_acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])

# Plot training accuracy
axs[0, 0].plot(train_acc, label='Training Accuracy', marker='o')
axs[0, 0].set_title('Training Accuracy')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()
axs[0, 0].grid()

# Plot training loss
axs[0, 1].plot(train_loss, label='Training Loss', marker='o', color='red')
axs[0, 1].set_title('Training Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[0, 1].grid()

# Plot validation accuracy (if available)
if val_acc:
    axs[1, 0].plot(val_acc, label='Validation Accuracy', marker='o', color='green')
    axs[1, 0].set_title('Validation Accuracy')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    axs[1, 0].grid()
else:
    axs[1, 0].text(0.5, 0.5, 'No Validation Accuracy Data', fontsize=12, ha='center')

# Plot validation loss (if available)
if val_loss:
    axs[1, 1].plot(val_loss, label='Validation Loss', marker='o', color='orange')
    axs[1, 1].set_title('Validation Loss')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    axs[1, 1].grid()
else:
    axs[1, 1].text(0.5, 0.5, 'No Validation Loss Data', fontsize=12, ha='center')

# Set the figure title
plt.suptitle('DeepFake Detector Training Performance', fontsize=16)

# Display the plots
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust title placement
plt.show()
