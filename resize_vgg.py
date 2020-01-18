# -*- encoding: utf-8 -*-
'''
@File    :   resize_vgg.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2018/12/25 15:18   xin      1.0         None
'''

from skimage.io import imread, imsave
from skimage.transform import resize

import glob


def resize_image(input_path, save_path, file_type='tif'):
    """
    图片resize，vgg输入要求224*224,故影像在输入网络之前需要resize为224*224
    :param input_path: 存放输入图片文件夹
    :param save_path: resize之后保存路径
    :param file_type: 文件类型
    :return:
    """
    images = glob.glob(input_path+"/*."+file_type)
    for image in images:
        file_name =image[image.rindex("\\")+1:]
        img = imread(image)
        new_img = resize(img, (224, 224))
        imsave(save_path+"/"+file_name.replace("tif", "png").replace("jpg", "png"), new_img)


if __name__ == "__main__":
    input_path = r'G:\xin.data\datasets\mlw\data\validating\img'
    save_path = r'G:\xin.data\datasets\mlw\data\validating\img_resize'
    resize_image(input_path, save_path, "tif")




