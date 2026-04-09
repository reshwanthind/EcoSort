import os
import warnings

# CLEAN TERMINAL & LOGGING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# FOLDER SETUP 
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# GPU MEMORY MANAGEMENT
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected and Memory Growth enabled.")
    except RuntimeError as e:
        print(f"!! GPU Error: {e}")

#  DATA PREPROCESSING 
DATASET_PATH = "dataset/Garbage classification"

BATCH_SIZE = 16         # increase to 32 if GPU allows
IMG_SIZE = (224, 224)   # increase to 300x300 if GPU allows

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,        
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',      
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

#  HIGH-ACCURACY GPU ARCHITECTURE 
def build_high_acc_model():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),

        # Block 1
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),

        # Block 2 
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),

        # Block 4 
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),

        layers.Flatten(),

        # Dense Layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        
        layers.Dense(6, activation='softmax') 
    ])

    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_high_acc_model()

# TRAINING WITH CALLBACKS 
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)

print("\n Training is starting...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[reduce_lr], # Added callback here
    verbose=1
)

# SAVE RESULTS 
model_path = os.path.join(RESULTS_DIR, "garbage_model_high_acc.h5")
model.save(model_path)
print(f"\nTraining complete. Model saved at: {model_path}")

# Plotting Accuracy and Loss Graphs
plt.figure(figsize=(14, 5))

# Accuracy Subplot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'ro', label='Training accuracy') # Red dots
plt.plot(history.history['val_accuracy'], 'g-', label='Validation accuracy') # Green line
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Subplot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'ro', label='Training loss')
plt.plot(history.history['val_loss'], 'g-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot
graph_path = os.path.join(RESULTS_DIR, "training_metrics.png")
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
print(f"Graphs saved at: {graph_path}")

plt.show()