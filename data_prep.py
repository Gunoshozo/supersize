import PIL
from PIL import Image
import  os,shutil

import os
from os import listdir
from os.path import isfile,join
import random

def downsize_im(image, output_size = (256, 256)):
    size = image.size
    if(is_square(size) and size[0] >= output_size[0]):
        im = image.resize(output_size)
    elif(size[0] >= output_size[0] and size[1] >= output_size[1]):
        return image.crop((0,0,output_size[0],output_size[1]))
    else:
        return image

def get_additional(image,os = (256,256)):
    size = image.size
    xs = size[0]/os[0]
    ys = size[1]/os[1]
    ims = []
    if xs < 1 or ys < 1:
        return []
    if xs < 2 and ys < 2:
        return [image.crop((0,0,256,256)),]
    for i in range(1,int(xs)):
        for j in range(int(ys)):
            #if i == int(xs):
            #    b = (size[0]-256,os[1] * j,size[0], os[1] * (j + 1))
            #else:
            b = (os[0] * i, os[1] * j, os[0] * (i + 1), os[1] * (j + 1))
            ims.append(image.crop(b))
    return ims


def res_n_save(path,size,save_path = '',transforms = (False,False,False,False)):
    names = get_files_list(path)
    pics = [Image.open(path+'/'+ n) for n in names]
    for i in range(len(pics)):
        tmp_pics = get_additional(pics[i],size)
        #if len(tmp_pics) != 1:
         #   tmp_pics[0] = downsize_im(pics[i],size)
        for j in range(len(tmp_pics)):
            if tmp_pics[j] != None:
                tmp_pics[j].save(save_path + '/' + str(j) + names[i]  , "JPEG")
                if tmp_pics[j].size == size:
                    tmp_pics[j].save(save_path + '/' + names[i], "JPEG")
                    if transforms[0] == True:
                        im = tmp_pics[j].transpose(Image.FLIP_LEFT_RIGHT)
                        im.save(save_path + '/' + 'flr' + names[i], "JPEG")
                    if transforms[1] == True:
                        im = tmp_pics[j].transpose(Image.FLIP_TOP_BOTTOM)
                        im.save(save_path + '/' + 'ftb' + names[i], "JPEG")
                    if transforms[2] == True:
                        im = tmp_pics[j].rotate(90)
                        im.save(save_path + '/' + 'r90' + names[i], "JPEG")
                    if transforms[3] == True:
                        im = tmp_pics[j].rotate(180)
                        im.save(save_path + '/' + 'r180' + names[i], "JPEG")


def resize_n_back(path,save_path = '',size = (256,256),scale = 1,names =[]):
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
                if j <= bs/3:
                    scale_size = (int(size[0] / scale), int(size[1] / scale))
                elif j > bs/3 and j <= bs*2/3:
                    scale_size = (int(size[0] / (scale * 2)), int(size[1] / (scale * 2)))
                else:
                    scale_size = (int(size[0] / (scale * 1.5)), int(size[1] / (scale * 1.5)))
                pics[j] = pics[j].resize(scale_size,PIL.Image.BILINEAR)
                pics[j] = pics[j].resize(size,Image.NEAREST)
                pics[j].save(save_path + '/' + names[n], "JPEG")
                n+=1
    else:
        pics = [Image.open(path + '/' + n) for n in names]
        for j in range(len(pics)):
            pics[j] = pics[j].resize(scale_size)
            pics[j] = pics[j].resize(size)
            pics[j].save(save_path + '/' + names[j], "JPEG")


def prepate_ims(foldername ,path, base_size = (256, 256), scale = 1,trans=(False,False,False,False)):
    #save base sized pics
    s_path1 ='base ' + foldername + ' ' + str(base_size[0]) +'x'+ str(base_size[1])
    s_path2 = 'downscaled ' + foldername + ' ' + str(int(base_size[0]/scale)) +'x'+ str(int(base_size[1]/scale)) + 'and'  + str(int(base_size[0]/(scale*2))) +'x'+ str(int(base_size[1]/(scale*2)))
    if not os.path.exists(s_path1):
        os.makedirs(s_path1)
    res_n_save(path, base_size, save_path=s_path1, transforms=trans)
    #save scaled pics
    if not os.path.exists(s_path2):
        os.makedirs(s_path2)
    resize_n_back(s_path1, s_path2, scale=2)


def missing_files(full_list_path,created_list_path):
    names1 = get_files_list(full_list_path)
    names2 = get_files_list(created_list_path)
    set1 = set(names1)
    set2 = set(names2)
    set1 = set1.difference(set2)
    if not len(set1) == 0:
        resize_n_back(full_list_path,created_list_path,scale=2,names=list(set1))
        return True
    else:
        return False

def delete_missing(full_list_path,created_list_path):
    names1 = get_files_list(full_list_path)
    names2 = get_files_list(created_list_path)
    set1 = set(names1)
    set2 = set(names2)
    set1 = set1.difference(set2)
    for p in set1:
        os.remove(full_list_path+''+p)



def split_files(path ='',path2 ='',percentage = 0.85):
    files = listdir(path)
    random.shuffle(files)
    if not os.path.exists(path+'_train'):
        os.makedirs(path+'_train')
    if not os.path.exists(path+'_valid'):
        os.makedirs(path+'_valid')

    if not os.path.exists(path2+'_train'):
        os.makedirs(path2+'_train')
    if not os.path.exists(path2+'_valid'):
        os.makedirs(path2+'_valid')

    for i in range(len(files)):
        if i <= len(files)*percentage:
            shutil.copy(path + '\\'+files[i],path+'_train')
            shutil.copy(path2 + '\\' + files[i], path2 + '_train')
        else:
            shutil.copy(path + '\\' + files[i], path + '_valid')
            shutil.copy(path2 + '\\' + files[i], path2 + '_valid')

def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: return False
    return True

def get_missing(full_list_path,created_list_path):
    names1 = get_files_list(full_list_path)
    names2 = get_files_list(created_list_path)
    set1 = set(names1)
    set2 = set(names2)
    set1 = set1.difference(set2)
    paths = [full_list_path + '/'+ n for n in list(set1)]
    if not os.path.exists(full_list_path +'_missing'):
        os.makedirs(full_list_path +'_missing')
    for i in range(len(paths)):
        im = Image.open(paths[i])
        im.save(full_list_path+'_missing/'+list(set1)[i])


def get_files_list(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files

def get_files_paths(path):
    files = [path + '\\' + f for f in listdir(path) if isfile(join(path, f))]
    return files


def is_square(size):
    if size[0] == size[1]:
        return True
    else:
        return False

def check_ims(path):
    names = get_files_paths(path)
    for p in names:
        if(is_grey_scale(p)):
            print(p)

if __name__ == '__main__':
    #prepate_ims(foldername='Dataset',path='Dataset',scale=2,trans=(True,True,True,False))
    #missing_files('base Dataset 256x256','downscaled Dataset 128x128and64x64')
    split_files(path='base Dataset 256x256',path2='downscaled Dataset 128x128and64x64')