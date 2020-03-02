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
from keras import layers
from keras.layers import Input, Add,  BatchNormalization,  Conv2D, ReLU,Dense,Flatten
from keras.models import Model, model_from_json,Sequential
from keras.optimizers import Adam,SGD
from keras.losses import mean_squared_error
from keras.callbacks import History
import matplotlib.pyplot as plt





hist = History()


Conv_num = 228
Shape = (256,256,3)


def SuperSizer(input_shape = (256,256,3), conv_l= Conv_num):
    X_input = Input(input_shape)

    shortcut = X_input

    y = Conv2D(128, (3, 3), padding='same')(X_input)
    y = ReLU()(y)

    for i in range(conv_l-1):
        y = Conv2D(128,(3,3),padding='same')(y)
        #y = BatchNormalization()(y)
        y = ReLU()(y)

    y = Conv2D(3, (3, 3), padding='same')(y)
    y = layers.Add()([shortcut,y])
    model = Model(inputs= X_input,outputs =y, name = 'SupSizer')
    return model

def SuperSizer228(input_shape = (512,512,3), conv_l= Conv_num):
    X_input = Input(input_shape)

    shortcut = X_input

    y = Conv2D(64,(3,3),padding='same')(X_input)
    y = BatchNormalization()(y)
    y = ReLU()(y)


    y = Dense(64,activation='relu',use_bias=True)(y)
    y = Dense(64, activation='relu',use_bias=True)(y)


    y = Conv2D(128, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    y = Dense(128, activation='relu', use_bias=True)(y)
    y = Dense(128, activation='relu', use_bias=True)(y)

    y = Conv2D(3, (3, 3), padding='same')(y)
    y = layers.Add()([shortcut,y])
    model = Model(inputs= X_input,outputs =y, name = 'SupSizer')
    return model

def SuperSizerU(input_shape = (512,512,3), conv_l= Conv_num):
    X_input = Input(input_shape)

    shortcut = X_input

    y = Conv2D(32,(3,3),padding='same')(X_input)
    y = ReLU()(y)

    shortcut32 = y


    y = Conv2D(32, (3, 3), padding='same')(y)
    y = Add()([shortcut32,y])
    y = ReLU()(y)

    y = Conv2D(64, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    shortcut64 = y


    y = Conv2D(64, (3, 3), padding='same')(y)
    y = Add()([shortcut64,y])
    y = ReLU()(y)

    y = Conv2D(128, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    shortcut128 = y

    y = Conv2D(128, (3, 3), padding='same')(y)
    y = Add()([shortcut128,y])
    y = ReLU()(y)

    y = Conv2D(3, (3, 3), padding='same')(y)
    y = Add()([shortcut,y])
    model = Model(inputs= X_input,outputs =y, name = 'SupSizer')
    return model


dirname = ' satellite '
train_path_X = 'downscaled'+dirname+'128x128_train'
train_path_Y = 'base'+dirname+'256x256_train'
valid_path_X = 'downscaled'+dirname+'128x128_valid'
valid_path_Y = 'base'+dirname+'256x256_valid'

def imageLoader(pathX, pathY, batch_size):


    file_list_X = get_files_paths(pathX)
    file_list_Y = get_files_paths(pathY)


    L = len(file_list_X)
    assert (len(file_list_X) == len(file_list_Y))
    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X1 =[np.array(Image.open(f).convert('RGB')) for f in file_list_X[batch_start:batch_end]]
            X = np.array(X1)
            Y1 =[np.array(Image.open(f).convert('RGB')) for f in file_list_Y[batch_start:batch_end]]
            Y = np.array(Y1)

            X = np.true_divide(X,255)
            Y = np.true_divide(Y,255)

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples

            batch_start = clamp(L,batch_start+batch_size)
            batch_end = clamp(L,batch_end+batch_size)

def rmse(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true),axis=-1))

def nrmse(y_true,y_pred):
    return -rmse(y_true,y_pred)

def clamp(max,val):
    if val > max:
        return max
    else:
        return val

def PSNR(y_true,y_pred):
    return tf.image.psnr(y_true,y_pred,1.0)

def assemble_and_train(shape,n = Conv_num,lr_=0.001,ep=1,batch_size = 20):

    train_samples = len(get_files_list(train_path_X))
    valid_samples = len(get_files_list(valid_path_X))
    ss_model = SuperSizer228(input_shape=shape, conv_l = n)
    opt = Adam(lr=lr_,clipvalue=0.001)
    ss_model.compile(opt, loss='mean_squared_error',metrics=[PSNR,rmse,'mae'])
    ss_model.summary()
    print("Assembled, now training")
    ss_model.fit_generator(generator=imageLoader(train_path_X,train_path_Y,batch_size), validation_data=imageLoader(valid_path_X,valid_path_Y,batch_size), steps_per_epoch=int(train_samples/batch_size), validation_steps=int(valid_samples/batch_size), epochs=ep, callbacks=[hist])

    model_json = ss_model.to_json()
    with open(str(n)+"model.json", "w") as json_file:
        json_file.write(model_json)
    ss_model.save_weights(str(n)+"model0.h5")
    print("Model have been saved to the disk")

def get_image(path,scale = 1.):
    X = Image.open(path)
    X = X.resize((int(X.size[0]*scale),int(X.size[1]*scale)))
    return np.array(X)

def predict(path,version,n):

    files = get_files_paths(path)
    names = get_files_list(path)
    Ims = [Image.open(n) for n in files]
    Arr = [np.array(i) for i in Ims]
    for i in range(len(Ims)):
        X = Arr[i]
        ss_model = SuperSizer228(input_shape=X.shape,conv_l=n)
        ss_model.load_weights(str(Conv_num)+'model' + str(version)+'.h5')
        X = np.expand_dims(X,axis=0)
        X = np.true_divide(X, 255)
        X3 = ss_model.predict(X)
        X3 = np.squeeze(X3,axis=0)
        X3 = np.multiply(X3,255)
        scipy.misc.toimage(X3,cmin=0,cmax=255).save('output\\' +names[i])


def test():
    t = get_files_paths(train_path_Y)
    for f in t:
        Image.open(f)
    t = get_files_paths(train_path_X)
    for  f in t:
        Image.open(f)


def train(lr_, ep = 1, n_of_run = 1, verb=1,a = 10,batch_size=15):
    train_samples = len(get_files_list(train_path_X))
    valid_samples = len(get_files_list(valid_path_X))
    json_file = open(str(Conv_num)+'model.json', 'r')
    ss_model_json = json_file.read()
    json_file.close()
    ss_model = model_from_json(ss_model_json)
    ss_model.load_weights(str(Conv_num)+'model'+str(n_of_run-1)+'.h5')
    opt = Adam(lr=lr_) #, clipvalue=0.00001
    #opt = SGD(lr=lr_, momentum=0.9, decay=1e-4, clipvalue=(1 / lr_))
    ss_model.compile(opt, loss='mean_squared_error', metrics=[PSNR,rmse,'mae','accuracy'])
    #keras.utils.plot_model(ss_model,to_file="Model.png")
    ss_model.fit_generator(generator=imageLoader(train_path_X,train_path_Y,batch_size), validation_data=imageLoader(valid_path_X,valid_path_Y,batch_size), steps_per_epoch=int(train_samples/batch_size), validation_steps=int(valid_samples/batch_size), epochs=ep, callbacks=[hist])
    ss_model.save_weights(str(Conv_num)+'model'+str(n_of_run)+'.h5')
    #return (hist.history['val_acc'][-1] + hist.history['acc'][-1]) / 2
    return -(hist.history['loss'][-1]+hist.history['val_loss'][-1])



def Upscale(path,save_path,sf =1,ver=1):
    files = get_files_paths(path)
    names = get_files_list(path)
    Ims = [Image.open(n) for n in files]
    Sizes = [(int(i.size[0] * sf),int(i.size[1] * sf)) for i in Ims]
    for i in range(len(Ims)):
        Ims[i] = Ims[i].resize(Sizes[i])
        Ims[i].save(save_path + '\\'+names[i])
    predict(save_path,ver,Conv_num)

def comp(Xpath,Ypath):
    X = Image.open(Xpath)
    X1 = np.array(X,dtype=np.float64)
    Y = Image.open(Ypath)
    Y1 = np.array(Y,dtype=np.float64)
    X1 = X1-Y1
    #Im = Image.fromarray(np.uint8(X1))
    #Im = PIL.ImageOps.invert(Im)
    #X1 = np.array(Im)
    scipy.misc.toimage(X1, cmin=0, cmax=255).save('diff.jpg')

def plot_loss():
    plt.plot(hist.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

def main():
    #Upscale(path='test\\Downscaled',save_path='test\\NN',sf=2,ver=1)
    #assemble_and_train(shape=Shape,lr_=0.1,ep=2,batch_size=8)
    train(ep=10, n_of_run=1, lr_=0.1,a=50,batch_size=8)

    #train(ep=10, n_of_run=2, lr_=0.01,a=50,batch_size=8)
    #train(ep=10, n_of_run=3, lr_=0.001,a=50,batch_size=8)
    #train(ep=10, n_of_run=4, lr_=0.0001,a=50,batch_size=8)
    #train(ep=10, n_of_run=5, lr_=0.00001, a=50,batch_size=8)
    #train(ep=10, n_of_run=6, lr_=0.000001, a=50,batch_size=8)
    #print(hist.history)
    #print(hist.history['rmse'])
    #train(10,n_of_run=3,_lr=1e-4)
    #Upscale('input',sf=1,ver=0)
    #Xpath = 'input_upscaled\\1.1.jpg'
    #Ypath = 'output\\1.1.jpg'
    #comp(Xpath,Ypath)
    plot_loss()



if __name__ == '__main__':
    main()