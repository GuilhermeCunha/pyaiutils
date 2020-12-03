import numpy as np
import matplotlib

def to_categorical(y, n_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not n_classes:
        n_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, n_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (n_classes,)
    categorical = np.reshape(categorical, output_shape)
    
    return categorical

def to_1d(data):
    return np.array(data).argmax(axis=1)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap