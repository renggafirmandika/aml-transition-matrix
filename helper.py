# load the data and check the shape of its attributes
import numpy as np

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

    # print('Dataset loaded with the following shapes:')
    # print('Xtr:', Xtr.shape)
    # print('Str:', Str.shape)
    # print('Xts:', Xts.shape)
    # print('Yts:', Yts.shape)

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