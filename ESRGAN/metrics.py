import tensorflow as tf
import keras.backend as K
import numpy as np
from PIL import Image

from data_prep import get_files_paths


def clamp(max, val):
    if val > max:
        return max
    else:
        return val


def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, 1.0)


def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def imageLoader(pathX, pathY, batch_size):
    file_list_X = get_files_paths(pathX)
    file_list_Y = get_files_paths(pathY)

    L = len(file_list_X)
    assert (len(file_list_X) == len(file_list_Y))

    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:

            limit = min(batch_end, L)
            X1 = [np.array(Image.open(f).convert('RGB')) for f in file_list_X[batch_start:batch_end]]
            X = np.array(X1)
            Y1 = [np.array(Image.open(f).convert('RGB')) for f in file_list_Y[batch_start:batch_end]]
            Y = np.array(Y1)

            X = np.true_divide(X, 255)
            Y = np.true_divide(Y, 255)


            yield (X, Y)

            batch_start = clamp(L, batch_start + batch_size)
            batch_end = clamp(L, batch_end + batch_size)
