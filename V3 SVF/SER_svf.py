# ser_svm_with_plots_static.py
# SER con SVM (RBF) + gráficos y matriz de confusión (sin argparse; todo en CONFIG)
# Requisitos: librosa, numpy, scikit-learn, joblib, soundfile, matplotlib, tqdm

import os
import numpy as np
import joblib
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, recall_score,
                             accuracy_score)

# ============== CONFIG ==============
DATASET_PATH = "dataset"        # <- subcarpetas por clase: DATASET/Clase1/*.wav, Clase2/*.wav, ...
OUT_DIR      = "./svm_results"           # <- carpeta donde se guardan gráficos y reportes
MODEL_PATH   = "ser_svm.joblib"          # <- nombre de archivo del modelo final dentro de OUT_DIR
SR           = 16000                     # sample rate para cargar
RANDOM_STATE = 42
CV_OUTER     = 5                         # folds para CV externa
param_grid = [
    {"pca__n_components": [None, 100, 150, 200], 
     "svc__kernel": ["rbf"],
     "svc__C": [0.5, 1, 3, 5, 10],
     "svc__gamma": ["scale", 0.01, 0.02, 0.03, 0.05]}#,
    # {"pca__n_components": [None, 100, 150, 200],
    #  "svc__kernel": ["linear"],
    #  "svc__C": [0.1, 0.5, 1, 3, 5, 10]},
    # {"pca__n_components": [None, 100, 150, 200],
    #  "svc__kernel": ["poly"],
    #  "svc__degree": [2, 3],
    #  "svc__C": [0.5, 1, 3, 5, 10],
    #  "svc__gamma": ["scale", 0.01, 0.03],
    #  "svc__coef0": [0, 0.5, 1.0]}
]

# =====================================

# ------------- Audio & Features -------------

def load_audio(path, sr=SR, mono=True, n_fft=1024):
    y, file_sr = sf.read(path, always_2d=False)
    if y.ndim > 1 and mono:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y=y, orig_sr=file_sr, target_sr=sr, res_type="kaiser_fast")
    y = y.astype(np.float32)
    maxabs = np.max(np.abs(y)) if y.size else 1.0
    if maxabs > 0:
        y = y / maxabs
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)), mode="reflect")
    return y, sr

def frame_features(y, sr, n_mfcc=40, hop_length=216, n_fft=1024, fmin=50, fmax=None,
                   use_chroma=False, use_contrast=False, use_tonnetz=False, use_f0=True):
    """
    Devuelve features por frame (T,F). Al final harás pooling temporal.
    n_mfcc: sube a 30-40 para SER. hop_length=216 (~13.5 ms @16k) = +resolución temporal.
    """
    # ------- Bloque MFCC -------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                hop_length=hop_length, n_fft=n_fft,
                                fmin=fmin, fmax=fmax)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    # ------- Bloque espectral/prosódico básico -------
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    sfl = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)

    mats = [mfcc, d1, d2, rms, zcr, sc, sbw, sro, sfl]

    # ------- Extras útiles para SER -------
    if use_chroma:
        # Chroma: energía por clase de pitch (12 bins). Captura armonicidad/timbre alto.
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mats.append(chroma)

    if use_contrast:
        # Spectral contrast: diferencia entre picos/valleys espectrales. Muy útil para “brillo” vs “opacidad”.
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mats.append(contrast)

    if use_tonnetz:
        # Tonnetz: relaciones armónicas (requiere chroma internamente). Útil si hay variación tonales.
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        mats.append(tonnetz)

    if use_f0:
        # Pitch (F0) con YIN: prosodia fundamental (entonación).
        # Se entrega como vector frameado; alineamos por hop_length.
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # Convertir a forma (1, T_f0) y remuestrear a rejilla de frames del hop_length:
        # Truco: creamos un vector temporal para f0 y reenmarcamos con padding.
        # Más simple: expandimos/recortamos por proporción de longitudes.
        # (Ajuste leve para no complicar el pipeline)
        T_ref = mfcc.shape[1]
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # Match temporal aproximado por interpolación lineal
        t_old = np.linspace(0, 1, num=f0.shape[0], endpoint=True)
        t_new = np.linspace(0, 1, num=T_ref,    endpoint=True)
        f0i = np.interp(t_new, t_old, f0)[None, :]  # (1,T_ref)
        mats.append(f0i)

    # ------- Alinear por tiempo y apilar -------
    T = min(M.shape[1] for M in mats)
    mats = [M[:, :T] for M in mats]

    # Apila como (F_total, T) -> luego transpones a (T, F_total)
    return np.vstack(mats).T


def pool_statistics(X_tf):
    X_tf = np.nan_to_num(X_tf, nan=0.0, posinf=0.0, neginf=0.0)
    stats = [X_tf.mean(0), X_tf.std(0)]
    for q in (10, 50, 90):
        stats.append(np.percentile(X_tf, q, axis=0))
    v = np.concatenate(stats, axis=0).astype(np.float32)
    return np.clip(v, -10.0, 10.0)

def extract_features(path, sr=SR):
    y, sr = load_audio(path, sr=sr)
    if len(y) < sr // 2:
        y = np.pad(y, (0, sr - len(y)), mode="reflect")
    return pool_statistics(frame_features(y, sr))

# ------------- Dataset -------------

def is_audio(p):
    return os.path.splitext(p)[1].lower() in {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}

def load_dataset(dataset_path):
    classes = sorted([d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))])
    label2id = {c:i for i,c in enumerate(classes)}
    X_paths, y = [], []
    for c in classes:
        for fn in sorted(os.listdir(os.path.join(dataset_path, c))):
            fp = os.path.join(dataset_path, c, fn)
            if is_audio(fp):
                X_paths.append(fp); y.append(label2id[c])
    return np.array(X_paths), np.array(y, dtype=int), classes

def build_feature_matrix(paths):
    feats = []
    for p in tqdm(paths, desc="Extrayendo features", ncols=80):
        feats.append(extract_features(p))
    return np.vstack(feats)

# ------------- Plots -------------

def plot_confusion(y_true, y_pred, class_names, out_dir, normalize=False, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title(f"Matriz de confusión{title_suffix} " + ("(normalizada)" if normalize else "(cruda)"))
    fig.tight_layout()
    fname = os.path.join(out_dir, f"confusion_matrix{'_norm' if normalize else ''}.png")
    fig.savefig(fname)
    plt.close(fig)
    return fname

def plot_f1_bars(y_true, y_pred, class_names, out_dir):
    f1c = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.bar(range(len(class_names)), f1c)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel("F1 (por clase)")
    ax.set_title("F1 por clase")
    fig.tight_layout()
    fname = os.path.join(out_dir, "f1_per_class.png")
    fig.savefig(fname)
    plt.close(fig)
    return fname

def plot_learning_curve(estimator, X, y, out_dir, title="Learning Curve (SVM RBF)"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="f1_macro", shuffle=True, random_state=RANDOM_STATE
    )
    train_mean, train_std = train_scores.mean(1), train_scores.std(1)
    val_mean, val_std = val_scores.mean(1), val_scores.std(1)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(train_sizes, train_mean, marker='o', label='Train')
    ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
    ax.plot(train_sizes, val_mean, marker='s', label='CV')
    ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)
    ax.set_xlabel("Tamaño de entrenamiento (muestras)")
    ax.set_ylabel("F1 macro")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(out_dir, "learning_curve.png")
    fig.savefig(fname)
    plt.close(fig)
    return fname

# ------------- Entrenamiento / Evaluación -------------

def nested_cv_and_plots(X, y, class_names, out_dir, random_state=RANDOM_STATE):
    os.makedirs(out_dir, exist_ok=True)
    base_pipe = make_pipeline(
        StandardScaler(),
        PCA(),
        SVC(class_weight="balanced", probability=False, random_state=random_state)
    )
    lc_path = plot_learning_curve(base_pipe, X, y, out_dir)

    outer = StratifiedKFold(n_splits=CV_OUTER, shuffle=True, random_state=random_state)

    y_true_all, y_pred_all = [], []
    accs, f1m_list, uar_list = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        gs = GridSearchCV(base_pipe, param_grid=param_grid, scoring="f1_macro",
                          n_jobs=-1, cv=inner, refit=True, verbose=0)
        gs.fit(X_tr, y_tr)
        y_pred = gs.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average="macro")
        uar = recall_score(y_te, y_pred, average="macro")

        accs.append(acc); f1m_list.append(f1m); uar_list.append(uar)
        y_true_all.extend(y_te.tolist()); y_pred_all.extend(y_pred.tolist())

        print(f"[Fold {fold}] best={gs.best_params_} | Acc={acc:.4f} F1m={f1m:.4f} UAR={uar:.4f}")

    def stat(a): return np.mean(a), np.std(a)
    acc_m, acc_s = stat(accs)
    f1m_m, f1m_s = stat(f1m_list)
    uar_m, uar_s = stat(uar_list)

    print(f"\nResumen CV {CV_OUTER}-fold -> Acc {acc_m:.4f}±{acc_s:.4f} | "
          f"F1-macro {f1m_m:.4f}±{f1m_s:.4f} | UAR {uar_m:.4f}±{uar_s:.4f}")

    cm_path  = plot_confusion(y_true_all, y_pred_all, class_names, out_dir, normalize=False, title_suffix=" (CV)")
    cmn_path = plot_confusion(y_true_all, y_pred_all, class_names, out_dir, normalize=True,  title_suffix=" (CV)")
    f1_path  = plot_f1_bars(y_true_all, y_pred_all, class_names, out_dir)

    report = classification_report(y_true_all, y_pred_all, target_names=class_names, digits=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
        f.write(f"ACC_mean={acc_m:.4f}±{acc_s:.4f}\n"
                f"F1_macro_mean={f1m_m:.4f}±{f1m_s:.4f}\n"
                f"UAR_mean={uar_m:.4f}±{uar_s:.4f}\n")

    return {"acc": (acc_m, acc_s), "f1m": (f1m_m, f1m_s), "uar": (uar_m, uar_s),
            "cm": cm_path, "cm_norm": cmn_path, "f1_bars": f1_path, "lc": lc_path}

def fit_full_and_save(X, y, model_path, random_state=RANDOM_STATE):
    pipe = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=random_state)
    )
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_macro",
                      n_jobs=-1, cv=inner, refit=True, verbose=1)
    gs.fit(X, y)
    out_path = os.path.join(OUT_DIR, model_path) if os.path.isdir(OUT_DIR) else model_path
    joblib.dump(gs.best_estimator_, out_path)
    print("Guardado:", out_path, " | best_params:", gs.best_params_)
    return gs.best_estimator_

# ------------- Run -------------

def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    X_paths, y, classes = load_dataset(DATASET_PATH)
    print(f"Clases: {classes} | Total audios: {len(X_paths)}")

    X = build_feature_matrix(X_paths)
    print("Shape features:", X.shape)

    # CV + gráficos
    results = nested_cv_and_plots(X, y, classes, OUT_DIR)

    # Entrena final con todo y guarda
    fit_full_and_save(X, y, MODEL_PATH)

    print("\nArchivos generados en:", os.path.abspath(OUT_DIR))
    print(" -", os.path.basename(results['cm']))
    print(" -", os.path.basename(results['cm_norm']))
    print(" -", os.path.basename(results['f1_bars']))
    print(" -", os.path.basename(results['lc']))
    print(" - classification_report.txt")
    print("Modelo:", os.path.join(os.path.abspath(OUT_DIR), os.path.basename(MODEL_PATH)))

if __name__ == "__main__":
    run()
