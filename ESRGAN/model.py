import numpy as np
from PIL.Image import Image
from keras.callbacks import History
from keras.layers import Input, Add, Conv2D, ReLU, Lambda, Deconvolution2D, Conv2DTranspose, LeakyReLU, \
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
import keras.backend as K

from ESRGAN.metrics import PSNR, rmse, ssim, imageLoader
from data_prep import get_files_list


class upscale_model():

    def __init__(self, low_size, sf, blocks, filters,save, lr,weight_location,verbose=False,increasing=False):
        self.low_size = low_size
        self.size = int(low_size * sf)
        self.blocks = blocks
        self.weight_location = weight_location
        if increasing:
            self.model = self.increasing_model(blocks=blocks, filters=filters, input_shape=(self.low_size, self.low_size, 3),
                                    _size=(self.size, self.size, 3))
        else:
            self.model = self.ups_model(blocks=blocks, filters=filters, input_shape=(self.low_size, self.low_size, 3),
                                    _size=(self.size, self.size, 3))
        self.assemble_model(lr=lr, blocks=blocks, verbose=verbose,save=save)
        self.history = History()
        self.opt = Adam(lr=lr,clipvalue=0.5)

        self.loss = 'mean_squared_error'

    block_count = 1


    def block(self,x, kernel, filters, beta):
        x_shortcut = x
        x = self.dense_block(x, kernel, filters, beta)
        x = self.dense_block(x, kernel, filters, beta)
        x = self.dense_block(x, kernel, filters, beta)
        x = Lambda(lambda x: x * beta)(x)
        x = Add(name="block_end_" + str(self.block_count))([x, x_shortcut])
        self.block_count += 1
        return x

    def dense_block(self,x, kernel, filters, beta):
        x_shortcut1 = x
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), padding="same")(x)
        x = ReLU()(x)
        x = Add()([x, x_shortcut1])
        x_shortcut2 = x
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), padding="same")(x)
        x = ReLU()(x)
        x = Add()([x, x_shortcut1, x_shortcut2])
        x_shortcut3 = x
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), padding="same")(x)
        x = ReLU()(x)
        x = Add()([x, x_shortcut1, x_shortcut2, x_shortcut3])
        x_shortcut4 = x
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), padding="same")(x)
        x = ReLU()(x)
        x = Add()([x, x_shortcut1, x_shortcut2, x_shortcut3, x_shortcut4])
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), padding="same")(x)
        x = Lambda(lambda t: t * beta)(x)
        x = Add()([x, x_shortcut1])
        return x

    def ups_model(self, blocks=3, filters=64, input_shape=(128,128,3), _size=(256,256,3), kernel=3, beta=0.2):
        x_input = Input(input_shape)
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), padding="same")(x_input)
        x_shortcut = x
        for i in range(blocks):
            x = self.block(x, kernel, filters, beta)
        #    x = BatchNormalization()(x)
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), padding="same")(x)
        x = Add()([x, x_shortcut])

        x = Conv2DTranspose(filters=filters,kernel_size=(2,2),strides=2,padding='valid')(x)
        x = Conv2D(filters=kernel, kernel_size=(kernel, kernel), padding="same")(x)
        x = Conv2D(filters=3, kernel_size=(kernel, kernel), padding="same")(x)
        model = Model(inputs=x_input, outputs=x, name='ESRGAN?')
        return model

    def increasing_model(self, blocks=3, filters=32, input_shape=(128,128,3), _size=(256,256,3), kernel=3, beta=0.2):
        x_input = Input(input_shape)
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), padding="same")(x_input)
        x_shortcut = x
        for i in range(blocks):
            filter = filters*(i+1)
            x = self.block(x, kernel, filter, beta)
        x = Conv2D(filters=filters, kernel_size=(kernel, kernel), padding="same")(x)
        x = Add()([x, x_shortcut])

        x = Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=2, padding='valid')(x)
        x = Conv2D(filters=kernel, kernel_size=(kernel, kernel), padding="same")(x)
        x = Conv2D(filters=3, kernel_size=(kernel, kernel), padding="same")(x)
        model = Model(inputs=x_input, outputs=x, name='ESRGAN?')
        return model

    def assemble_model(self, lr, blocks, save, verbose=True,loss='mean_squared_error'):
        opt = Adam(lr=lr,clipvalue=0.5)
        self.model.compile(opt, loss=loss, metrics=[PSNR, ssim, rmse, 'mae'])
        if verbose:
            self.model.summary()
            plot_model(self.model, to_file="model" + str(self.blocks) + ".png")
        if save:
            self.model.save(self.weight_location +str(self.blocks) + ' model ' +'0.h5')



    def insert_layer_before_upscale(self, kernel, filters, startnumber,number, beta=0.2):
        index = 1+47*startnumber
        x = self.model.layers[index].output
        for i in range(number):
            self.blocks +=1
            x = self.block(x, kernel, filters, beta)
        x = Conv2D(filters=filters,kernel_size=(kernel, kernel), padding="same")(x)
        x = self.model.layers[-6]([x,self.model.layers[1].output])
        x = Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=2, padding='valid')(x)
        x = Conv2D(filters=3, kernel_size=(kernel, kernel), padding="same")(x)
        x = Conv2D(filters=3, kernel_size=(kernel, kernel), padding="same")(x)
        self.model = Model(inputs=self.model.layers[0].output, outputs=x, name='ESRGAN?')
        self.model.compile(self.opt, loss=self.loss, metrics=[PSNR, ssim, rmse, 'mae'])
        self.model.summary()

    def freeze_layers(self,num):
        total = 1 + 47*num
        for i in range(1,total+1):
            if self.model.layers[i] != self.model.layers[-6]:
                self.model.layers[i].trainable = False
        self.model.compile(self.opt, loss=self.loss, metrics=[PSNR, ssim, rmse, 'mae'])
        self.model.summary()

    def freeze(self,n,m):
        l = 1+47*n
        r = 1+47*m
        for i in range(l,r+1):
            if self.model.layers[i] != self.model.layers[-6]:
                self.model.layers[i].trainable = False
        self.model.compile(self.opt, loss=self.loss, metrics=[PSNR, ssim, rmse, 'mae'])
        self.model.summary()

    def set_paths(self, train_path_X, valid_path_X, train_path_Y, valid_path_Y):
        self.train_path_X = train_path_X
        self.valid_path_X = valid_path_X
        self.train_path_Y = train_path_Y
        self.valid_path_Y = valid_path_Y

    def train(self, batch_size, ep):
        train_samples = len(get_files_list(self.train_path_X))
        valid_samples = len(get_files_list(self.valid_path_X))
        self.model.fit_generator(generator=imageLoader(self.train_path_X, self.train_path_Y, batch_size=batch_size)
                                 , validation_data=imageLoader(self.valid_path_X, self.valid_path_Y, batch_size=batch_size)
                                 , steps_per_epoch=int(train_samples / batch_size)
                                 , validation_steps=int(valid_samples / batch_size),
                                 epochs=ep, callbacks=[self.history])


    def load_weights(self,weight_file):
        self.model.load_weights(weight_file)

    def save_weights(self,weight_file):
        self.model.save_weights(weight_file)

    def predict(self, sample:Image):
        x = np.array(sample)
        x = np.expand_dims(x, axis=0)
        x = np.true_divide(x, 255)
        y = self.model.predict(x)
        y = np.square(y,axis = 0)
        y = np.array(np.multiply(y,255),dtype=np.int)
        return y
