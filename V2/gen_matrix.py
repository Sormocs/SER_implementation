import numpy as np
import matplotlib.pyplot as plt

# --- Datos proporcionados ---
cm = np.array([
    [228, 26, 23, 11, 39, 58],
    [32, 301, 29, 8, 13, 1],
    [30, 35, 220, 36, 56, 8],
    [29, 16, 37, 199, 24, 80],
    [16, 10, 27, 5, 267, 35],
    [26, 1, 4, 18, 47, 288]
])

classes = ["disgusto", "enojo", "feliz", "miedo", "neutral", "triste"]

# --- Graficar matriz de confusión ---
plt.figure(figsize=(7, 6), dpi=120)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, ha="right")
plt.yticks(tick_marks, classes)

# Etiquetas de cada celda
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix_custom.png", dpi=150)
plt.show()
