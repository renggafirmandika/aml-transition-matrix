# load the data and check the shape of its attributes
import pandas as pd
import numpy as np
from itertools import product
from cnn_model import cnn_model
from loss_functions import (
    symmetric_cross_entropy,
    forward_correction_loss,
    CoTeachingProxyLoss,
    RememberRateScheduler,
    _infer_noise_rate_from_name, 
)
import anchor_estimator
import tensorflow as tf
from tensorflow import keras

RANDOM_SEED = 42

BEST_PARAMS = {
    ("FashionMNIST0.3", "sce"):       {"alpha": 0.01, "beta": 1.0, "A": -1.0, "lr": 0.001},
    ("FashionMNIST0.6", "sce"):       {"alpha": 0.01, "beta": 0.5, "A": -4.0, "lr": 0.001},
    ("CIFAR",           "sce"):       {"alpha": 0.05, "beta": 1.0, "A": -4.0, "lr": 0.001},

    ("FashionMNIST0.3", "coteaching"):{"warmup": 10, "lr": 0.001},
    ("FashionMNIST0.6", "coteaching"):{"warmup": 5,  "lr": 0.0005},
    ("CIFAR",           "coteaching"):{"warmup": 5,  "lr": 0.0005},
}


def load_dataset(filepath:str, dataset_name:str):
    dataset = np.load(filepath)
    Xtr = dataset['Xtr']
    Str = dataset['Str']
    Xts = dataset['Xts']
    Yts = dataset['Yts']

    T = None
    if dataset_name == 'FashionMNIST0.3':
        T = np.array([
            [0.7, 0.3, 0.0],
            [0.0, 0.7, 0.3],
            [0.3, 0.0, 0.7]
        ])
    elif dataset_name == 'FashionMNIST0.6':
        T = np.array([
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4]
        ])
        
    if len(Xtr.shape) == 2:
        Xtr = Xtr.reshape(-1, 28, 28, 1)
        Xts = Xts.reshape(-1, 28, 28, 1)

    Str = Str.astype(np.int32).ravel()
    Yts = Yts.astype(np.int32).ravel()

    return Xtr, Str, Xts, Yts, T

def split_data(X, y, train_ratio=0.8, random_seed=None):
    n_samples = len(X)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    return X_train, y_train, X_val, y_val

def _make_grid(param_space):
    keys = list(param_space.keys())
    for values in product(*[param_space[k] for k in keys]):
        yield {k: v for k, v in zip(keys, values)}

def _val_accuracy(model, X_val, y_val):
    # [loss, acc]; we want accuracy
    _, acc = model.evaluate(X_val, y_val, verbose=0)
    return float(acc)

def tune_hyperparams(Xtr, Str, dataset, method, input_shape,num_classes=3, n_dev_runs=3, epochs=30):

    # Normalization
    Xtr = Xtr.astype('float32') / 255.0

    # Define small, sensible grids
    if method == "sce":
        param_space = {
            "alpha": [0.01, 0.05, 0.1],
            "beta":  [0.5, 1.0],
            "A":     [-1.0, -2.0, -4.0],
            "lr":    [1e-3]  # keep optimizer stable unless you really need to explore
        }
    elif method == "coteaching":
        param_space = {
            "warmup": [3, 5, 10],
            "lr":     [1e-3, 5e-4]
        }
    else:
        raise ValueError(f"No tuner defined for method={method}")

    best_params, best_score = None, -np.inf

    for params in _make_grid(param_space):
        val_scores = []
        for r in range(n_dev_runs):
            seed = RANDOM_SEED + 1000 + r  # separate from main runs
            X_tr, y_tr, X_va, y_va = split_data(Xtr, Str, train_ratio=0.8, random_seed=seed)

            if method == "sce":
                model = train_model(
                    X_tr, y_tr, X_va, y_va,
                    dataset=dataset, method="sce",
                    epochs=epochs, input_shape=input_shape, num_classes=num_classes,
                    sce_params=params
                )
            elif method == "coteaching":
                model = train_model(
                    X_tr, y_tr, X_va, y_va,
                    dataset=dataset, method="coteaching",
                    epochs=epochs, input_shape=input_shape, num_classes=num_classes,
                    coteaching_params=params
                )

            acc = _val_accuracy(model, X_va, y_va)
            val_scores.append(acc)

            # cleanup
            del model
            tf.keras.backend.clear_session()

        mean_val_acc = float(np.mean(val_scores))
        print(f"[TUNE] dataset={dataset} method={method} params={params} -> val_acc={mean_val_acc:.4f}")

        if mean_val_acc > best_score:
            best_score = mean_val_acc
            best_params = dict(params)

    print(f"[TUNE][BEST] dataset={dataset} method={method} -> {best_params} (val_acc={best_score:.4f})")
    return best_params

def train_model(X_train, y_train, X_val, y_val,dataset, method="fc",transition_matrix=None,epochs=50, input_shape=(28, 28, 1), num_classes=3,sce_params=None, coteaching_params=None, learning_rate=0.001):
    model = cnn_model(input_shape=input_shape, num_classes=num_classes)
    callbacks = []

    if method in {"ce", "baseline"}:                        
        loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    elif method == "sce":
        # Defaults (your original)
        if dataset == "FashionMNIST0.3":
            alpha, beta = 0.01, 1.0
        elif dataset == "FashionMNIST0.6":
            alpha, beta = 0.01, 1.0
        elif dataset == "CIFAR":
            alpha, beta = 0.1, 1.0
        else:
            alpha, beta = 0.1, 1.0
        A = -4.0

        # Overrides from tuning
        if sce_params is not None:
            alpha = float(sce_params.get("alpha", alpha))
            beta  = float(sce_params.get("beta",  beta))
            A     = float(sce_params.get("A",     A))
            learning_rate = float(sce_params.get("lr", learning_rate))

        loss_function = symmetric_cross_entropy(alpha=alpha, beta=beta, A=A, num_classes=num_classes)

    elif method in {"fc", "forward"}:
        # if transition_matrix is None:
        #     transition_matrix = anchor_estimator.estimate_T(dataset, 10, 1)
        # assert transition_matrix is not None, "FC requires a (known or estimated) transition matrix."
        loss_function = forward_correction_loss(transition_matrix, num_classes=num_classes)

    elif method == "coteaching":
        loss_obj = CoTeachingProxyLoss(remember_rate=1.0, num_classes=num_classes)
        loss_function = loss_obj
        noise_rate = _infer_noise_rate_from_name(dataset)
        warmup = 5
        if coteaching_params is not None:
            warmup = int(coteaching_params.get("warmup", warmup))
            learning_rate = float(coteaching_params.get("lr", learning_rate))
        rr_sched = RememberRateScheduler(loss_obj, max_epochs=epochs, noise_rate=noise_rate, warmup=warmup)
        callbacks.append(rr_sched)

    else:
        raise ValueError(f"Unknown method: {method}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=['accuracy']
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=128,
        callbacks=[early_stopping, *callbacks],
        verbose=0
    )
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_test) * 100
    return accuracy

def _get_best_params(dataset, method):
    return BEST_PARAMS.get((dataset, method), None)

def run_single_experiment(Xtr, Str, Xts, Yts, T, dataset, method,num_runs=10, epochs=50, method_params=None):
    Xtr = Xtr.astype('float32') / 255.0
    Xts = Xts.astype('float32') / 255.0
    input_shape = Xtr.shape[1:]
    num_classes = 3
    if method in {"sce", "coteaching"} and method_params is None:
        method_params = _get_best_params(dataset, method)
    
    # FC path left as-is
    if method in {"fc", "forward"}:
        transition_matrix = T  
    else:
        transition_matrix = None

    accuracies = []
    for run in range(num_runs):
        seed = RANDOM_SEED + run
        X_train, y_train, X_val, y_val = split_data(
            Xtr, Str, train_ratio=0.8, random_seed=seed
        )

        if method == "sce":
            model = train_model(
                X_train, y_train, X_val, y_val,
                dataset=dataset, method="sce",
                transition_matrix=transition_matrix,
                epochs=epochs, input_shape=input_shape, num_classes=num_classes,
                sce_params=method_params  # <--- tuned params
            )
        elif method == "coteaching":
            model = train_model(
                X_train, y_train, X_val, y_val,
                dataset=dataset, method="coteaching",
                transition_matrix=transition_matrix,
                epochs=epochs, input_shape=input_shape, num_classes=num_classes,
                coteaching_params=method_params  # <--- tuned params
            )
        elif method in {"fc", "forward"}:
            model = train_model(
                X_train, y_train, X_val, y_val,
                dataset=dataset, method="fc",
                transition_matrix=transition_matrix,
                epochs=epochs, input_shape=input_shape, num_classes=num_classes
            )
        elif method in {"ce", "baseline"}:
            model = train_model(
                X_train, y_train, X_val, y_val,
                dataset=dataset, method="ce",
                epochs=epochs, input_shape=input_shape, num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        accuracy = evaluate_model(model, Xts, Yts)
        accuracies.append(accuracy)
        print(f"Run {run+1}/{num_runs}: Test Accuracy = {accuracy:.2f}%")

        del model
        tf.keras.backend.clear_session()

    return accuracies

def run_all_experiments(datasets, methods, num_runs=10, epochs=50):
    results = []
    
    for dataset in datasets:
        data_path = f'datasets/{dataset}.npz'
        Xtr, Str, Xts, Yts, T = load_dataset(data_path, dataset) 
        if dataset == 'CIFAR':
            T = anchor_estimator.estimate_T(dataset, 10, 0)
        for method in methods:
            print(f"Running {method.upper()} on {dataset}...")

            accuracies = run_single_experiment(
                Xtr, Str, Xts, Yts, T, dataset, method, num_runs, epochs
            )
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            results.append({
                'Dataset': dataset,
                'Method': method.upper(),
                'Mean': mean_acc,
                'Std': std_acc,
                'Result': f"{mean_acc:.2f} ± {std_acc:.2f}"
            })

            print(f"Result: {mean_acc:.2f} ± {std_acc:.2f}%")
    
    results_df = pd.DataFrame(results)
    
    return results_df