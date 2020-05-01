from PIL import Image,ImageDraw,ImageFont
from random import randint
import  itertools
from data_prep import *

font_list = ['TrajanPro-Regular.otf','arial.ttf','GARA.TTF','BOD_R.TTF']

def create_image(directory,tex,fontsize,text_count):
    randint(0,len(font_list))
    s = font_list[randint(0,len(font_list)-1)]
    font = ImageFont.truetype(s,fontsize)
    clr = ()
    for i in range(0,3):
        clr = clr + (randint(0,255),)
    img = Image.new('RGB',(256,256),clr)
    d = ImageDraw.ImageDraw(img)
    neg_clr = (~clr[0]& 0xFF,~clr[1]&0xFF,~clr[2]&0xFF)
    for i in range(0,text_count):
        text_pos = (randint(0, 200), randint(0, 200))
        d.text(xy=text_pos,font= font,text=tex,fill=neg_clr)
    img.save(directory +'\\' + tex+'.jpg');

def generate_strings(length,lownup=True,digts=True):
    if lownup:
        if digts:
            s = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        else:
            s = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    else:
        if digts:
            s = " abcdefghijklmnopqrstuvwxyz1234567890"
        else:
            s = s = " abcdefghijklmnopqrstuvwxyz"
    return ["".join(item) for item in itertools.product(s,repeat= length)]

def delete_files(deleteprs=.0,path=''):
    paths = get_files_paths(path)
    partial_size =int(len(paths)*deleteprs)
    for i in range(partial_size):
        index = randint(0,len(paths))
        os.remove(paths[index])
        paths.remove(paths[index])

def main():
    #prepate_ims('testims','test',(512,512),2)
    missing_files('base testims 256x256','downscaled testims 128x128)
    #split_files('base testims 256x256','downscaled testims 128x128and64x64')
    #seq = generate_strings(length=3,lownup=False)
    #seq.remove('aux')
    #seq.remove('prn')
    #seq.remove('con')
    #seq.remove('nul')
    #for string in seq:
    #    font_size = randint(20, 60)
    #    text_count = randint(10, 15)
    #    create_image('test',string,font_size,text_count)


if __name__ =='__main__':
    main()