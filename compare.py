import numpy
import math
from PIL import Image

dirname = ' Dataset '
imname = '0__ada_wong_resident_evil_drawn_by_liang_xing__sample-a77a07ab31dcdc93b72550faa44dbed9.jpg'
path_a = 'downscaled'+dirname+'128x128and64x64_train' +'/'+imname
path_b = 'base'+dirname+'256x256_train' + '/' + imname



def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    Before = Image.open(path_a)
    After = Image.open(path_b)
    print(psnr(numpy.array(Before),numpy.array(After)))
