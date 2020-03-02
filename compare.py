import numpy
import math
from PIL import Image
from data_prep import *



origs =  'test\\Origs'
downs =  'test\\Downscaled'
path_a = 'test\\Blerp'
path_b = 'test\\NN'




def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def downscale(path,save_path = '',size = (256,256),scale = 1,names =[]):
    if len(names) == 0:
        names = get_files_list(path)
    scale_size = (int(size[0] / scale), int(size[1] / scale))
    n = 0
    bs =50
    if(len(names)>bs):
        for p in range(int(round(len(names)/bs, 1))):
            if p == 0:
                ran =(0,(p+1)*bs)
            elif (p+1)*bs > len(names):
                ran = (p*bs+1,len(names)-2)
            else:
                ran = (p*bs+1,(p+1)*bs)
            pics = [Image.open(path + '/' + n) for n in names[ran[0]:ran[1]+1]]
            for j in range(len(pics)):
                #if j <= bs/3:
                scale_size = (int(size[0] / scale), int(size[1] / scale))
                #elif j > bs/3 and j <= bs*2/3:
                #    scale_size = (int(size[0] / (scale * 2)), int(size[1] / (scale * 2)))
                #else:
                #    scale_size = (int(size[0] / (scale * 1.5)), int(size[1] / (scale * 1.5)))
                pics[j] = pics[j].resize(scale_size,Image.BILINEAR)
                pics[j].save(save_path + '/' + names[n], "JPEG")
                n+=1

def comp(path):
    files_blerp = get_files_paths(path)
    files_origs = get_files_paths(origs)
    psnrs = []
    for i in range(len(files_blerp)):
        imb = np.array(Image.open(files_blerp[i]))
        imo = np.array(Image.open(files_origs[i]))
        psnrs.append(psnr(imo,imb))
    return  psnrs




if __name__ == '__main__':
    psnr_blerp=comp(path_a)
    psnr_nn=comp(path_b)
    print(psnr_blerp)
    print('_________________\n')
    print((psnr_nn))
    print(np.argmax(np.abs(np.subtract(psnr_nn,psnr_blerp))))
    #downscale(origs,downs,scale=2)
    #resize_n_back(origs,path_a,scale=2)
    #list_origs = get_files_paths(origs)
    #list_blerp = get_files_paths(path_a)
    #list_NN = get_files_paths(path_b)
    #get_missing(origs,list_blerp)
    #get_missing(origs, list_blerp)
#