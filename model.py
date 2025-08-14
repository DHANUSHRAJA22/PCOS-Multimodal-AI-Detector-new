# model.py
# Trains a VGG16-based classifier and saves the best model as:
#   models/pcos_detector_158.h5
# It also writes the class order to:
#   models/pcos_detector_158.labels.txt

import os
import json
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "train_directory2"     # directory with 2 subfolders (your classes)
IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 10                 # fine-tuning
BEST_MODEL_PATH = "models/pcos_detector_158.h5"
LABELS_PATH = "models/pcos_detector_158.labels.txt"
SEED = 123

os.makedirs("models", exist_ok=True)

# -----------------------------
# Datasets (with split)
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

# Save/print class order to avoid label mismatches at inference
class_names = train_ds.class_names
print("Class order (index -> label):")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

# Performance pipeline: normalize, cache, prefetch
AUTOTUNE = tf.data.AUTOTUNE

def normalize(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

train_ds = (train_ds
            .map(normalize, num_parallel_calls=AUTOTUNE)
            .cache()
            .shuffle(1000)
            .prefetch(AUTOTUNE))

val_ds = (val_ds
          .map(normalize, num_parallel_calls=AUTOTUNE)
          .cache()
          .prefetch(AUTOTUNE))

# -----------------------------
# Data augmentation
# -----------------------------
data_augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.10),
], name="augment")

# -----------------------------
# Model (VGG16 base + head)
# -----------------------------
base = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # Stage 1: freeze

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augment(inputs)
x = base(x, training=False)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(2, activation="softmax")(x)  # 2 classes

model = keras.Model(inputs, outputs, name="pcos_detector_158")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

# -----------------------------
# Train (Stage 1: frozen base)
# -----------------------------
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks,
)

# -----------------------------
# Fine-tune (Stage 2: unfreeze top layers)
# -----------------------------
# Unfreeze last few convolutional blocks for a small fine-tune
base.trainable = True
# Optionally, keep most layers frozen, only unfreeze last ~4 layers
for layer in base.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks,
)

print(f"\nBest model saved to: {BEST_MODEL_PATH}")
print(f"Class order written to: {LABELS_PATH}")
