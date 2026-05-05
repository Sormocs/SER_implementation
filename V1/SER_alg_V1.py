import os
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

tf.config.threading.set_intra_op_parallelism_threads(4) 
tf.config.threading.set_inter_op_parallelism_threads(4)

# ==============================
# CONFIGURACIONES
# ==============================
DATASET_PATH = "dataset"  # carpeta con subcarpetas por clase
SAMPLE_RATE = 20000
DURATION = 6  # segundos
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 30

# ==============================
# FUNCIÓN PARA CARGAR AUDIO
# ==============================
def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Padding o truncado
    if len(signal) > SAMPLES_PER_FILE:
        signal = signal[:SAMPLES_PER_FILE]
    else:
        pad_width = SAMPLES_PER_FILE - len(signal)
        signal = np.pad(signal, (0, pad_width), mode="constant")

    # Extraer MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T  # transpuesta → (tiempo, coeficientes)
    return mfcc

# ==============================
# CARGAR DATASET COMPLETO
# ==============================
def load_dataset(dataset_path):
    X, y = [], []
    class_names = os.listdir(dataset_path)
    label2id = {name: i for i, name in enumerate(class_names)}

    for label in class_names:
        folder = os.path.join(dataset_path, label)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            mfcc = load_audio(file_path)
            X.append(mfcc)
            y.append(label2id[label])

    return np.array(X, dtype=object), np.array(y), class_names

print("Cargando dataset...")
X, y, class_names = load_dataset(DATASET_PATH)

# ==============================
# AJUSTAR DIMENSIONES (padding a MFCCs)
# ==============================
# max_len = max(x.shape[0] for x in X)
# X_pad = np.array([np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode="constant") for x in X])

max_len = max(x.shape[0] for x in X)
X_pad = np.array(
    [np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode="constant") for x in X],
    dtype=np.float32  # <- importante
)

y = np.array(y, dtype=np.int32)

# ==============================
# SEPARAR TRAIN/TEST
# ==============================
X_train, X_val, y_train, y_val = train_test_split(X_pad, y, test_size=0.2, stratify=y, random_state=42)

# ==============================
# CREAR TENSORES CON tf.data
# ==============================
def preprocess(x, y):
    x = tf.expand_dims(x, -1)  # (tiempo, mfcc, 1)
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(preprocess).shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==============================
# MODELO CNN
# ==============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len, N_MFCC, 1)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==============================
# ENTRENAMIENTO
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==============================
# GRAFICAR RESULTADOS
# ==============================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title("Precisión")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title("Función de pérdida")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()

plt.show()


