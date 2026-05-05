# -*- coding: utf-8 -*-
"""
Full audio CNN pipeline (STREAMING VERSION):
- Loads by file path (no X in RAM)
- MFCC(+Δ,+ΔΔ)
- Train-only normalization (from sample)
- SpecAugment
- Regularization + callbacks
"""

import os
import glob
import random
import json
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# =====================================
# Reproducibility & TF threading
# =====================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =====================================
# CONFIG
# =====================================
DATASET_PATH = "dataset"        # Each subfolder = class
SAMPLE_RATE  = 16000
DURATION     = 4                # seconds
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
N_MFCC       = 60               # base MFCCs; we will add deltas
ADD_DELTAS   = False             # include Δ and ΔΔ
BATCH_SIZE   = 8
EPOCHS       = 60
LEARNING_RATE= 1e-2
WEIGHT_DECAY = 1e-4
USE_CLASS_WEIGHTS = True
MODEL_OUT    = "audio_cnn_model.keras"

# SpecAugment config (applied on TRAIN only)
TIME_MASK_PCT     = 0.10
FREQ_MASK_PCT     = 0.10
NUM_TIME_MASKS    = 2
NUM_FREQ_MASKS    = 2

# =====================================
# Helpers
# =====================================
def is_audio_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"]

def load_audio(file_path: str) -> np.ndarray:
    """Load, trim silence, pre-emphasize, pad/truncate, return MFCC(+Δ,+ΔΔ) as (T, F)."""
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    signal, _ = librosa.effects.trim(signal, top_db=30)
    if signal.size > 1:
        signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    if len(signal) > SAMPLES_PER_FILE:
        signal = signal[:SAMPLES_PER_FILE]
    else:
        signal = np.pad(signal, (0, SAMPLES_PER_FILE - len(signal)), mode="constant")

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC).astype(np.float32)  # (n_mfcc, T)
    feats = [mfcc]
    if ADD_DELTAS:
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feats.extend([delta, delta2])
    feats = np.vstack(feats).T  # -> (T, F)
    return feats

def spec_augment_np(x: np.ndarray) -> np.ndarray:
    """SpecAugment in NumPy (optional – we’ll use TF version below if desired)."""
    return x  # placeholder if you want to keep all aug in TF

# =====================================
# Label space
# =====================================
print("Scanning dataset...")
class_names = sorted([d for d in os.listdir(DATASET_PATH)
                      if os.path.isdir(os.path.join(DATASET_PATH, d))])
label2id = {name: i for i, name in enumerate(class_names)}
print("Classes:", class_names)
if not class_names:
    raise RuntimeError("No class folders found in DATASET_PATH.")

# =====================================
# Build file path list (no MFCCs yet)
# =====================================
file_paths, labels = [], []
for label in class_names:
    folder = os.path.join(DATASET_PATH, label)
    for fname in sorted(os.listdir(folder)):
        fp = os.path.join(folder, fname)
        if is_audio_file(fp):
            file_paths.append(fp)
            labels.append(label2id[label])

if not file_paths:
    raise RuntimeError("No audio files found. Check DATASET_PATH and formats.")

file_paths = np.array(file_paths)
labels     = np.array(labels, dtype=np.int32)

# =====================================
# Split by PATH (not tensors)
# =====================================
p_train, p_val, y_train, y_val = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# =====================================
# Train-only normalization stats (from a sample of train files)
# =====================================
def sample_paths(arr, k=400):
    if len(arr) <= k:
        return arr
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(arr), size=k, replace=False)
    return arr[idx]

def compute_norm_stats(sampled_paths):
    sums = None
    sqrs = None
    count = 0
    F = None
    for p in sampled_paths:
        x = load_audio(p)  # (T, F)
        if F is None:
            F = x.shape[1]
            sums = np.zeros(F, dtype=np.float64)
            sqrs = np.zeros(F, dtype=np.float64)
        sums += x.sum(axis=0)
        sqrs += (x**2).sum(axis=0)
        count += x.shape[0]
    mean = (sums / max(count, 1)).astype(np.float32)      # (F,)
    var  = (sqrs / max(count, 1) - mean**2).astype(np.float32)
    std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std

mean_F, std_F = compute_norm_stats(sample_paths(p_train, k=400))
FEATS = mean_F.shape[0]
print("Feature dims (F):", FEATS)

with open("norm_stats.json", "w") as f:
    json.dump({"mean": mean_F.tolist(), "std": std_F.tolist()}, f)

# =====================================
# tf.data streaming pipelines
# =====================================
# (A) SpecAugment in TF
def spec_augment_tf(x):
    # x: (T, F)
    T = tf.shape(x)[0]
    F = tf.shape(x)[1]

    def time_mask_once(x_):
        max_width = tf.maximum(tf.cast(tf.math.round(TIME_MASK_PCT * tf.cast(T, tf.float32)), tf.int32), 1)
        w = tf.random.uniform([], 0, max_width + 1, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, tf.maximum(T - w, 1), dtype=tf.int32)
        mask = tf.concat([tf.ones((t0, F), tf.float32),
                          tf.zeros((w,  F), tf.float32),
                          tf.ones((T - t0 - w, F), tf.float32)], axis=0)
        return x_ * mask

    def freq_mask_once(x_):
        max_width = tf.maximum(tf.cast(tf.math.round(FREQ_MASK_PCT * tf.cast(F, tf.float32)), tf.int32), 1)
        w = tf.random.uniform([], 0, max_width + 1, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, tf.maximum(F - w, 1), dtype=tf.int32)
        mask = tf.concat([tf.ones((T, f0), tf.float32),
                          tf.zeros((T, w), tf.float32),
                          tf.ones((T, F - f0 - w), tf.float32)], axis=1)
        return x_ * mask

    for _ in range(NUM_TIME_MASKS):
        x = time_mask_once(x)
    for _ in range(NUM_FREQ_MASKS):
        x = freq_mask_once(x)
    return x

# (B) Python loader wrapped for TF (load → normalize → (optional) augment → expand dims)
def load_feats_py(path_bytes):
    path = path_bytes.decode("utf-8")
    x = load_audio(path).astype(np.float32)          # (T, F)
    x = (x - mean_F) / std_F                         # normalize per feature
    return x

# def tf_load_feats_train(path, y):
#     x = tf.numpy_function(load_feats_py, [path], tf.float32)  # (T, F)
#     x.set_shape([None, FEATS])
#     x = spec_augment_tf(x)                     # augment only on train
#     x = tf.expand_dims(x, -1)                  # (T, F, 1)
#     return x, y

# def tf_load_feats_val(path, y):
#     x = tf.numpy_function(load_feats_py, [path], tf.float32)
#     x.set_shape([None, FEATS])
#     x = tf.expand_dims(x, -1)
#     return x, y

def tf_load_feats(path, y):
    x = tf.numpy_function(load_feats_py, [path], tf.float32)  # (T, F)
    x.set_shape([None, FEATS])
    x = tf.expand_dims(x, -1)                                 # (T, F, 1)
    return x, y

def tf_augment(x, y):
    x = tf.squeeze(x, -1)             # (T, F)
    x = spec_augment_tf(x)            # apply masks
    x = tf.expand_dims(x, -1)         # back to (T, F, 1)
    return x, y

options = tf.data.Options()
options.experimental_deterministic = True

# train_ds = (tf.data.Dataset.from_tensor_slices((p_train, y_train))
#             .shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
#             .map(tf_load_feats_train, num_parallel_calls=1)
#             .padded_batch(BATCH_SIZE, padded_shapes=([None, FEATS, 1], []), drop_remainder=False)
#             .prefetch(1)
#             .with_options(options))

# val_ds = (tf.data.Dataset.from_tensor_slices((p_val, y_val))
#           .map(tf_load_feats_val, num_parallel_calls=1)
#           .padded_batch(BATCH_SIZE, padded_shapes=([None, FEATS, 1], []))
#           .prefetch(1)
#           .with_options(options))

# train_ds = train_ds.cache("cache_train.tfcache")
# val_ds   = val_ds.cache("cache_val.tfcache")

# ---------- TRAIN ----------
train_base = (tf.data.Dataset.from_tensor_slices((p_train, y_train))
              .map(tf_load_feats, num_parallel_calls=1)
              .cache("cache_train.tfcache"))    # cache deterministic tensors to DISK

train_ds = (train_base
            .map(tf_augment, num_parallel_calls=1)  # fresh aug each epoch (not cached)
            .shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
            .padded_batch(BATCH_SIZE, padded_shapes=([None, FEATS, 1], []), drop_remainder=False)
            .prefetch(1)
            .with_options(options))

# ---------- VAL ----------
val_ds = (tf.data.Dataset.from_tensor_slices((p_val, y_val))
          .map(tf_load_feats, num_parallel_calls=1)
          .cache("cache_val.tfcache")           # safe to cache, no aug/shuffle
          .padded_batch(BATCH_SIZE, padded_shapes=([None, FEATS, 1], []))
          .prefetch(1)
          .with_options(options))

print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val   batches:", tf.data.experimental.cardinality(val_ds).numpy())

# =====================================
# Model (Conv -> BN -> ReLU) x3 + GAP + Dropout + Dense
# =====================================
from tensorflow.keras import layers, regularizers, models

wd = WEIGHT_DECAY
inp = layers.Input(shape=(None, FEATS, 1))   # NOTE: variable time now

# 32→24, 64→48, 128→96
x = layers.Conv2D(24, (3,3), padding='same', kernel_regularizer=regularizers.l2(wd))(inp)
x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(48, (3,3), padding='same', kernel_regularizer=regularizers.l2(wd))(x)
x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(96, (3,3), padding='same', kernel_regularizer=regularizers.l2(wd))(x)
x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(wd))(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# =====================================
# Callbacks
# =====================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.CSVLogger("training_log.csv", append=False),
]

# =====================================
# Optional class weights for imbalance
# =====================================
class_weight = None
if USE_CLASS_WEIGHTS:
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Class weights:", class_weight)

# =====================================
# Train
# =====================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=2
)

# =====================================
# Evaluate & Diagnostics
# =====================================
print("\nEvaluating on validation set...")
val_metrics = model.evaluate(val_ds, verbose=0)
print(f"Validation — Loss: {val_metrics[0]:.4f} | Accuracy: {val_metrics[1]:.4f}")

# Predictions for metrics
y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
print("\nClassification report:")
print(classification_report(y_val, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_val, y_pred)
print("Confusion matrix:\n", cm)

# =====================================
# Plots
# =====================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.tight_layout()
plt.show()

# =====================================
# Save label mapping for inference
# =====================================
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump({"class_names": class_names, "label2id": label2id}, f, ensure_ascii=False, indent=2)

print(f"\nSaved model to: {MODEL_OUT}")
print("Saved normalization stats: norm_stats.json")
print("Saved labels: labels.json")
