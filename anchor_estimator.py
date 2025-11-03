import numpy as np
EPS = 1e-12

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + EPS)

def temperature_scale_probs(probs, y_int, T_grid=None):
    """
    the temperature makes the gap of softmax more reliable and more real
    """
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

def estimate_T_anchor_from_probs(probs, top_quantile=0.99, min_count=1):
    """
    Anchor point/extreme value estimation:
    For each class j, take the set of samples where p_j >= quantile threshold, and average their complete probability vectors as the j-th column of T;
    Finally, perform column normalization (column sum = 1).
    """
    N, C = probs.shape
    T_cols = []
    for j in range(C):
        pj = probs[:, j]
        thr = np.quantile(pj, top_quantile)
        idx = np.where(pj >= thr)[0]
        if idx.size < min_count:                      
            idx = np.array([int(np.argmax(pj))])
        col = probs[idx].mean(axis=0)                 
        T_cols.append(col)
    T = np.stack(T_cols, axis=1)                      
    T = np.clip(T, 0.0, None)
    T = T / (np.sum(T, axis=0, keepdims=True) + EPS)  
    return T
