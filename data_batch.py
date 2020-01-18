# -*- encoding: utf-8 -*-
'''
@File    :   data_batch.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/4/18 14:49   xin      1.0         None
'''

import numpy as np
import cv2
import glob
import itertools


from config import color_key_dic, cls_num_key_dic


color_key_dic = color_key_dic["wf_small"]
cls_num_key_dic = cls_num_key_dic["wf_small"]



def getImageArr(path, width, height, imgNorm="sub_mean"):
    try:
        img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":
            img = np.float32(img) / 127.5 - 1
        elif imgNorm == "sub_mean":
            # img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
        elif imgNorm == "divide":
            # img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img = img / 255.0

        # if odering == 'channels_first':
        #     img = np.rollaxis(img, 2, 0)
        return img
    except Exception as e:
        print (path, e)
        img = np.zeros((height, width, 3))
        # if odering == 'channels_first':
        #     img = np.rollaxis(img, 2, 0)
        return img


def getSegmentationArr(path, nClasses, width, height):
    seg_labels = np.zeros((height, width, nClasses))
    try:
        img = cv2.imread(path, 1)
        # img = cv2.resize(img, (width, height))
        # img = img[:, :, 0]
        for row in img:
            for cell in row:
                new_cell = color_key_dic[tuple(cell)]["cls_num"]

                for c in range(nClasses):
                    seg_labels[:, :, c] = (new_cell == c).astype(int)
    except Exception as e:
        print(e)

    # seg_labels = np.reshape(seg_labels, (width * height, nClasses))
    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height,
                               output_width):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()


    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):
        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = zipped.next()
            X.append(getImageArr(im, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)


if __name__ == "__main__":
    x = getSegmentationArr(r"G:\xin.data\rs\mlw\fcn\FCN_sample\training\gray_gt\5.png", 13, 224, 224)
    img = np.reshape(x,(224,224,13))
    img = np.argmax(img, 2)
    cv2.imwrite("test.png", img)