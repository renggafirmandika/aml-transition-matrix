import numpy as np
import helper
import cnn_model
import itertools
from tensorflow import keras


# EPS just avoid some of the situation like log(0)
EPS = 1e-12
RANDOM_SEED = 42

def softmax(z):
    '''
    it subtracts each row’s maximum value, and normalizes by the row sum.
    The result is a probability distribution over 𝐶 classes for each sample.
    '''
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + EPS)

def temperature_scale_probs(probs, y_int, T_grid=None):
    """it searches over a grid of temperature values and selects
    the temperature that minimizes the validation cross-entropy loss.

    Args:
        probs (np.ndarray): Predicted probabilities of shape [N, C] from the model.
        y_int (np.ndarray): Integer-encoded observed (noisy) labels of shape [N].
        T_grid (np.ndarray, optional): Array of candidate temperature values (default: np.linspace(0.5, 5.0, 30)).

    Returns:
        tuple
        probs_cal : np.ndarray
            Calibrated probabilities of shape [N, C].
        best_T : float
            Optimal temperature selected from the grid.
    """     
    if T_grid is None:
        T_grid = np.linspace(0.5, 5.0, 30)
    N, C = probs.shape
    logits = np.log(np.clip(probs, EPS, 1.0))
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
    """it estimates the noise transition matrix using the high-confidence (anchor) method.

    Args:
        probs : np.ndarray
            Calibrated probabilities of shape [N, C].
        y_noisy : np.ndarray
            Observed (noisy) labels of shape [N]. (Unused in this version.)
        top_k : int, optional
            Number of top high-confidence samples per class (default: 50).

    Returns:
        np.ndarray
            Estimated column-stochastic transition matrix of shape [C, C].
    """    
    N, C = probs.shape
    cols = []
    for j in range(C):
        conf_j = probs[:, j] 
        k = min(top_k, conf_j.size)
        top_idx = np.argpartition(conf_j, -k)[-k:] 
        anchors = probs[top_idx] 
        col = anchors.mean(axis=0).astype(np.float32)
        cols.append(col)
    T = np.stack(cols, axis=0)                      
    T = ensure_column_stochastic(T, EPS) 
    return T

# To match the anchor estimator calculation, make sure it aligns with the correct one
def ensure_column_stochastic(T: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    T = np.clip(T, 0, None)
    colsum = T.sum(axis=0, keepdims=True) + eps
    return T / colsum

def ensure_row_stochastic(T: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    T = np.clip(T, 0.0, None)
    rowsum = T.sum(axis=1, keepdims=True) + eps
    return T / rowsum

# How to calculate T
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

    # Normalization
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

    if T_true is not None:
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

        if T_true is not None:
            print(f"Mean Absolute Error:  {mae:.4f}")
            print(f"Max Absolute Error:   {max_error:.4f}")
            print(f"Frobenius Norm Error: {frobenius:.4f}")

    return T_hat
    