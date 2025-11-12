"""
PCB DEFECT DETECTION - TRAINING SCRIPT
Untuk Windows dengan CUDA 11.2 + cuDNN 8.1
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import json
from datetime import datetime

print("="*70)
print("PCB DEFECT DETECTION - TRAINING")
print("="*70)
print(f"TensorFlow: {tf.__version__}")
sys.stdout.flush()

# GPU Setup
print("\n[1/6] Checking GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Found: {gpus[0].name}")
        print("Memory growth: Enabled")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU detected - using CPU")
sys.stdout.flush()

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 200
DATASET_DIR = 'dataset'

print(f"\n[2/6] Loading MobileNetV2...")
sys.stdout.flush()

# Load pre-trained model
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully")
sys.stdout.flush()

print(f"\n[3/6] Creating data generators...")
sys.stdout.flush()

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load data
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Classes: {train_gen.class_indices}")
sys.stdout.flush()

print("\n[4/6] Testing data loading...")
test_x, test_y = next(iter(train_gen))
print(f"First batch shape: {test_x.shape}")
sys.stdout.flush()

print(f"\n[5/6] Starting training...")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
print("="*70)
sys.stdout.flush()

# Callbacks
callbacks = [
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    TensorBoard(
        log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        histogram_freq=0
    )
]

print("Starting model.fit()...")
sys.stdout.flush()

try:
    # Train with safe settings for Windows + ImageDataGenerator
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        workers=0,  # Main thread only (prevents hang on Windows)
        use_multiprocessing=False,
        max_queue_size=1
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    
    # Save final model
    print("\n[6/6] Saving model...")
    model.save('qc_inspector_model.h5')
    
    # Save training history
    with open('training_history.json', 'w') as f:
        history_dict = {key: [float(val) for val in values] 
                       for key, values in history.history.items()}
        json.dump(history_dict, f, indent=2)
    
    # Print results
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"\nFiles saved:")
    print(f"  - qc_inspector_model.h5")
    print(f"  - best_model.h5")
    print(f"  - training_history.json")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user!")
    print("Checkpoint saved in: best_model.h5")
    sys.exit(0)
    
except Exception as e:
    print(f"\nERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
