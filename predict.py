from config import cls_num_key_dic
from skimage.io import imread, imsave
from fcn8 import FCN8
import numpy as np
import glob

cls_num_key_dic = cls_num_key_dic["wf_small"]


def predict(img_file, save_path):
    fcn8 = FCN8()
    model = fcn8.get_model()
    model.load_weights(r"G:/xin.data/models/fcn/fcn8.hdf5")
    print('load_model success')
    img = imread(img_file).astype("float32")
    file_name = img_file[img_file.rindex("\\")+1:]

    img /= 255
    imgs_mask_predict = model.predict(np.array([img]), batch_size=1, verbose=1)

    covert_to_img(imgs_mask_predict[0], save_path, file_name)


def covert_to_img(img, save_path, filename):
    new_img = []
    for row in img:
        new_row = []
        for cell in row:
            cls = cell.argmax()
            new_cell = cls_num_key_dic[cls]["color"]
            new_row.append(new_cell)
        new_img.append(new_row)
    img = np.array(new_img)
#    final_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    imsave(save_path + "/{0}".format(filename), img)


if __name__ == "__main__":
    imgs = glob.glob(r'G:/xin.data/datasets/mlw/data/validating/img_resize/*.png')
    for img in imgs:
        predict(img, r'G:/xin.src/python/fcn/data/results')