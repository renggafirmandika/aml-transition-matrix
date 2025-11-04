import numpy as np
import helper
import cnn_model
import itertools
from tensorflow import keras

EPS = 1e-12
RANDOM_SEED = 42

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + EPS)

def temperature_scale_probs(probs, y_int, T_grid=None):
    if T_grid is None:
        T_grid = np.linspace(0.5, 5.0, 30)
    N, C = probs.shape
    logits = np.log(np.clip(probs, EPS, 1.0))   # softmax(logits)=probs
    y_one = np.eye(C)[y_int.astype(int)]

    def ce(p):
        return -np.mean(np.sum(y_one * np.log(np.clip(p, EPS, 1.0)), axis=1))

    best_T, best_loss = 1.0, ce(probs)
    for T in T_grid:
        p_scaled = softmax(logits / T)
        loss = ce(p_scaled)
        if loss < best_loss:
            best_T, best_loss = float(T), loss
    return softmax(logits / best_T), best_T

def estimate_T_anchor_from_probs(probs, y_noisy: np.ndarray, top_k: int = 50):
    N, C = probs.shape
    T_rows = []
    for i in range(C):
        idx_i = np.where(y_noisy == i)[0]
        if idx_i.size == 0:
            # fallback: uniform row if no examples with noisy label i
            T_rows.append(np.full((C,), 1.0 / C, dtype=np.float32))
            continue

        # confidence for class i on those examples
        conf_i = probs[idx_i, i]
        if idx_i.size <= top_k:
            top_idx = idx_i
        else:
            # pick top_k by confidence
            top_local = np.argpartition(conf_i, -top_k)[-top_k:]
            top_idx = idx_i[top_local]

        anchors = probs[top_idx]                # (K, C) ≈ p(tilde | X anchors of class i)
        row = anchors.mean(axis=0)              # empirical P(tilde=. | Y=i) up to norm
        T_rows.append(row.astype(np.float32))
    T = np.stack(T_rows, axis=0)                      
    T = ensure_row_stochastic(T, EPS) 
    return T


def ensure_column_stochastic(T: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    T = np.clip(T, 0, None)
    colsum = T.sum(axis=0, keepdims=True) + eps
    return T / colsum

def ensure_row_stochastic(T: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    T = np.clip(T, 0.0, None)
    rowsum = T.sum(axis=1, keepdims=True) + eps
    return T / rowsum

def estimate_T(dataset_name="", 
                           epochs=10, verbose=1):
    if verbose == 1:
        print(f"Processing: {dataset_name}")

    dataset_path = f"./datasets/{dataset_name}.npz"
    
    # Load data
    Xtr, Str, Xts, Yts, T_true = helper.load_dataset(dataset_path, dataset_name)
    
    # Preprocess
    Str = Str.astype("int64")
    Yts = Yts.astype("int64")

    Xtr = Xtr.astype("float32") / 255.0
    Xts = Xts.astype("float32") / 255.0

    X_tr, y_tr, X_val, y_val = helper.split_data(Xtr, Str, train_ratio=0.8, random_seed=7)

    num_classes = int(np.max(Str)) + 1
    input_shape = Xtr.shape[1:]

    m = cnn_model.cnn_model(input_shape=input_shape, num_classes=num_classes)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    m.fit(X_tr, y_tr,
      validation_data=(X_val, y_val),
      epochs=epochs, batch_size=128, verbose=verbose)

    p_val = m.predict(X_val, batch_size=128, verbose=0)              
    p_val_cal, bestT = temperature_scale_probs(p_val, y_val)          
    T_hat = estimate_T_anchor_from_probs(p_val_cal, y_noisy=y_val, top_k=50)  

    mae = np.mean(np.abs(T_hat - T_true))
    max_error = np.max(np.abs(T_hat - T_true))
    frobenius = np.linalg.norm(T_hat - T_true, 'fro')

   

    if verbose==1:
        print("Estimated Transition Matrix (T_hat):")

        print(np.round(T_hat, 2))                       
        print("------------------------------")

        print("True Transition Matrix (T_true):")

        print(T_true)                        
        print("------------------------------")

        print(f"Mean Absolute Error:  {mae:.4f}")
        print(f"Max Absolute Error:   {max_error:.4f}")
        print(f"Frobenius Norm Error: {frobenius:.4f}")

    return T_hat
    