# load the data and check the shape of its attributes
import numpy as np

def load_and_inspect(filepath:str):
    dataset = np.load(filepath)
    Xtr = dataset['Xtr']
    Str = dataset['Str']
    Xts = dataset['Xts']
    Yts = dataset['Yts']
    print('Dataset loaded with the following shapes:')
    print('Xtr:', Xtr.shape)
    print('Str:', Str.shape)
    print('Xts:', Xts.shape)
    print('Yts:', Yts.shape)

    return Xtr, Str, Xts, Yts