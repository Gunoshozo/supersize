import numpy as np
import  scipy.misc

from data_prep import get_files_paths,get_files_list
import keras
import time
import PIL
import PIL.ImageOps
import tensorflow as tf
from PIL import Image
import numpy as np
import keras.backend as K
from tensorflow.python.ops.image_ops_impl import psnr
from keras import layers
from keras.layers import Input, Add,  BatchNormalization,  Conv2D, ReLU,Dense,Flatten
from keras.models import Model, model_from_json,Sequential
from keras.optimizers import Adam,SGD
from keras.losses import mean_squared_error
from keras.callbacks import History
import matplotlib.pyplot as plt

conv_num = 5;
Input_shape = (128,128,3)
Output_shape =(256,256,3)

def Upscaler(pre_conv ,post_conv):
    X = Input(shape=Input_shape,)

    y = Conv2D(32,(3,3),padding='same',activation='relu')(X)

    for i in range(pre_conv-1):
        y = Conv2D(32, (3, 3), padding='same', activation='relu')(y)



    for i in range(post_conv ):
        y = Conv2D(32, (3, 3), padding='same', activation='relu')(y)
