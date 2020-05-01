import keras
from keras.optimizers import Adam

from data_prep import get_files_list
from ESRGAN.model import *
from ESRGAN.metrics import *
from keras.callbacks import History
from keras.utils import plot_model
from keras.engine.training import Model

train_path_X = '../x satellite 128x128_train'
train_path_Y = '../y satellite 256x256_train'
valid_path_X = '../x satellite 128x128_valid'
valid_path_Y = '../y satellite 256x256_valid'

blocks = 20
shape = (256, 256, 3)
filters = 64
kernel_size = 3

hist = History()


def assemble_model(input_shape, n, lr):
    model = upscale_model.ups_model(blocks=n, filters=filters, input_shape=input_shape, kernel=kernel_size)
    opt = Adam(lr=lr)
    model.compile(opt, loss='mean_squared_error', metrics=[PSNR, rmse, 'mae', ssim])
    model.summary()
    plot_model(model, to_file="model.png")
    model.save("model" + str(blocks) + '.h5')
    return model


def train(model: Model, batch_size, ep, n_of_run=1):
    train_samples = len(get_files_list(train_path_X))
    valid_samples = len(get_files_list(valid_path_X))
    model.fit_generator(generator=imageLoader(train_path_X, train_path_Y, batch_size=batch_size)
                        , validation_data=imageLoader(valid_path_X, valid_path_Y, batch_size=batch_size)
                        , steps_per_epoch=int(train_samples / batch_size)
                        , validation_steps=int(valid_samples / batch_size),
                        epochs=ep, callbacks=[hist])
    model.save_weights(str(blocks) + 'model' + str(n_of_run) + '.h5')

def predict(model, weight_file, sample:Image.Image,scale_factor):
    output_size = sample.size[0]*scale_factor



def main():
    assemble_model(shape, blocks, 0.0001)


if __name__ == '__main__':
    main()
