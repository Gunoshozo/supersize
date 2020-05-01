# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from PIL.Image import Image
from ESRGAN.model import upscale_model





def load_image(path):
    return Image.open(path)

dirname = ' satellite '
train_path_X = '../x'+dirname+'128x128_train'
train_path_Y = '../y'+dirname+'256x256_train'
valid_path_X = '../x'+dirname+'128x128_valid'
valid_path_Y = '../y'+dirname+'256x256_valid'

weight_location = './3 blocks/64 filters/5 bs/'

def main():
    nn = upscale_model(low_size=128, sf=2, blocks = 3, filters=64, lr=1e-3, verbose=True, save=True,weight_location=weight_location)
    run = 0


    weight_file =weight_location +str(nn.blocks) + ' model ' + str(run) +'.h5'
    nn.load_weights(weight_file)
    #nn.freeze_layers(9)
    #nn.freeze(0,3)
    #nn.insert_layer_before_upscale(kernel=3,filters=64,startnumber=9,number=2)
    nn.set_paths(train_path_X=train_path_X,train_path_Y=train_path_Y,valid_path_X=valid_path_X,valid_path_Y=valid_path_Y)
    nn.train(batch_size=5, ep=5)
    weight_file =weight_location +str(nn.blocks) + ' model ' + str(run+1) +'.h5'
    nn.save_weights(weight_file)
    if False:
        path = ".png"
        image = load_image(path)
        nn.predict()


if __name__ == '__main__':
    main()
