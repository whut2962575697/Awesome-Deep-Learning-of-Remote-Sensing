# -*- encoding:utf-8 -*-

import shutil
import glob
import numpy as np
import cv2
from skimage.io import imread
import xlwt
from matplotlib import pyplot as plt
from config import color_key_dic, cls_num_key_dic


np.random.seed(2)
color_key_dic = color_key_dic["wf_small"]
cls_num_key_dic = cls_num_key_dic["wf_small"]


def move_data(txt, src, dst):
    """
    :param txt:按照文本文档读取
    :param src:源目录
    :param dst:目标目录
    :return:
    """
    with open(txt, "r") as f:
        lines = f.readlines()
    for line in lines:
        da = line.strip("\n").split(" ")
        name = da[0][da[0].rindex("/")+1:]
        shutil.copy(src+"/"+name, dst+"/"+name)


def class_sample():
    """
    将样本按照训练集、验证集、测试集分类
    :return:
    """
    img_path = "data/big_scale/img"  # 所有样本图片集合路径
    label_path = "data/big_scale/label"  # 所有样本标签集合路径
    train_img_path = "data/big_scale/train_img"  # 用于训练样本图片集合路径
    train_label_path = "data/big_scale/train_label"  # 用于训练样本标签集合路径
    test_img_path = "data/big_scale/test_img"  # 用于测试样本图片
    test_label_path = "data/big_scale/test_label"  # 测试集用于对比结果
    valid_img_path = "data/big_scale/valid_img"  # 用于验证样本图片集合
    valid_label_path = "data/big_scale/valid_label"   # 用于训练样本标签集合路径
    move_data("data/train3.txt", img_path, train_img_path)
    move_data("data/train3.txt", label_path, train_label_path)
    move_data("data/val3.txt", img_path, valid_img_path)
    move_data("data/val3.txt", label_path, valid_label_path)
    move_data("data/testing.txt", img_path, test_img_path)
    move_data("data/testing.txt", label_path, test_label_path)


def split_sample(gt_path, image_path, train_image_path, train_label_path, val_image_path, val_label_path):
    gt_files = glob.glob(gt_path+"/*.png")
    image_files = glob.glob(image_path+"/*.png")
    if len(gt_files) != len(image_files):
        print(u"样本和标签数量不一致")
        return
    else:
        sample_num = len(gt_files)
        sample_list = np.arange(sample_num)
        np.random.shuffle(sample_list)
        for i in sample_list[:int(sample_num*0.75)]:
            gt_filename = gt_files[i][gt_files[i].rindex("\\")+1:]
            image_filename = image_files[i][image_files[i].rindex("\\")+1:]
            shutil.copy(image_files[i], train_image_path + "/" + image_filename)
            shutil.copy(gt_files[i], train_label_path + "/" + gt_filename)

        for i in sample_list[int(sample_num*0.75):]:
            gt_filename = gt_files[i][gt_files[i].rindex("\\")+1:]
            image_filename = image_files[i][image_files[i].rindex("\\")+1:]
            shutil.copy(image_files[i], val_image_path + "/" + image_filename)
            shutil.copy(gt_files[i], val_label_path + "/" + gt_filename)


def create_train_data(train_img_path, train_label_path,
                      npy_path, row_num, column_num, file_type, is_valid_data=False):
    """
    创建训练样本数据矩阵
    :param train_img_path: 训练集图片路径
    :param train_label_path: 训练集标签路径
    :param npy_path: 保存矩阵文件路径
    :param row_num: 图片的height
    :param column_num: 图片的width
    :param file_type: 文件类型
    :param is_valid_data: 是否为valid集
    :return:
    """
    print('-' * 30)
    print('creating train data...')
    print('-' * 30)
    imgs = glob.glob(train_img_path+"/*."+file_type)
    img_data = np.ndarray((len(imgs), 224, 224, 3), dtype=np.uint8)
    label_data = np.ndarray((len(imgs), row_num, column_num, 13), dtype=np.uint8)
    for i, _img in enumerate(imgs):
        filename = _img[_img.rindex("\\")+1:]

        img = imread(train_img_path+"/"+filename)
        label = imread(train_label_path+"/"+filename.replace("tif", "png"))
        img_data[i] = img
        new_label = []
        for row in label:
            new_row = []
            for cell in row:
                new_cell = color_key_dic[tuple(cell)]["cls"]

                new_row.append(new_cell)
            new_label.append(new_row)
        label_data[i] = new_label
    if is_valid_data:
        save_img = "valid_img_np.npy"
        save_label = "valid_label_np.npy"
    else:
        save_img = "train_img_np.npy"
        save_label = "train_label_np.npy"
    np.save(npy_path+"/"+save_img, img_data)
    np.save(npy_path+"/"+save_label, label_data)


def create_tes_data(test_img_path, npy_path, row_num, column_num, file_type):
    """
    创建测试数据矩阵
    :param test_img_path: 测试数据图片路径
    :param npy_path: 保存矩阵文件路径
    :param row_num: 图片的height
    :param column_num: 图片的width
    :param file_type: 文件类型
    :return:
    """
    print('-' * 30)
    print('creating test data...')
    print('-' * 30)
    imgs = glob.glob(test_img_path+"/*."+file_type)
    img_data = np.ndarray((len(imgs), 224, 224, 3), dtype=np.uint8)
    for i, _img in enumerate(imgs):
        filename = _img[_img.rindex("\\") + 1:]

        img = imread(test_img_path + "/" + filename)
        img_data[i] = img
    np.save(npy_path + "/test_img_np.npy", img_data)


def create_tes_label(cls_num, test_label_path, npy_path, row_num, column_num, file_type):
    print('-' * 30)
    print('creating test label...')
    print('-' * 30)
    labels = glob.glob(test_label_path+"/*."+file_type)
    labels_data = np.ndarray((len(labels), row_num, column_num, 1), dtype=np.uint8)
    for i, _label in enumerate(labels):
        label = imread(_label)
        new_label = []
        for row in label:
            new_row = []
            for cell in row:
                new_cell = color_key_dic[tuple(cell)]["cls_num"]
                if new_cell == cls_num:
                    new_cell = [1]
                else:
                    new_cell = [0]
                new_row.append(new_cell)
            new_label.append(new_row)
        labels_data[i] = new_label
    np.save(npy_path+"/test_label_np.npy", labels_data)


def convert_to_imgs(npy_path, save_path):
        print("array to image")
        imgs = np.load(npy_path+"/valid_label_np.npy")
        for i, img in enumerate(imgs):
            new_img = []
            for row in img:
                new_row = []
                for cell in row:

                    if cell[0] >= 0.5:
                        cls = 1
                    else:
                        cls = 0
                    new_cell = cls_num_key_dic[cls]["color"]
                    new_row.append(new_cell)
                new_img.append(new_row)
            img = np.array(new_img)
            final_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path+"/{0}.tif".format(i), final_img)


def calculate_accuracy(predict_labels, ture_labels):
    background_cell_total = 0
    road_cell_total = 0
    residence_cell_total = 0
    industry_cell_total = 0
    greenland_cell_total = 0
    uncompleteland_cell_total = 0
    forest_cell_total = 0
    playground_cell_total = 0
    water_cell_total = 0
    village_cell_total = 0
    service_cell_total = 0
    farmland_cell_total = 0
    others_cell_total = 0
    background_true_num = 0
    road_true_num = 0
    residence_true_num = 0
    industry_true_num = 0
    greenland_true_num = 0
    uncompleteland_true_num = 0
    forest_true_num = 0
    playground_true_num = 0
    water_true_num = 0
    village_true_num = 0
    service_true_num = 0
    farmland_true_num = 0
    others_true_num = 0
    for i, label in enumerate(predict_labels):
        for j, row in enumerate(label):
            for k, cell in enumerate(row):
                predict_cls = cell.argmax()
                true_cls = ture_labels[i][j][k].argmax()
                if true_cls == 0:
                    background_cell_total += 1
                    if predict_cls == 0:
                        background_true_num += 1
                elif true_cls == 1:
                    road_cell_total += 1
                    if predict_cls == 1:
                        road_true_num += 1
                elif true_cls == 2:
                    residence_cell_total += 1
                    if predict_cls == 2:
                        residence_true_num += 1
                elif true_cls == 3:
                    industry_cell_total += 1
                    if predict_cls == 3:
                        industry_true_num += 1
                elif true_cls == 4:
                    greenland_cell_total += 1
                    if predict_cls == 4:
                        greenland_true_num += 1
                elif true_cls == 5:
                    uncompleteland_cell_total += 1
                    if predict_cls == 5:
                        uncompleteland_true_num += 1
                elif true_cls == 6:
                    forest_cell_total += 1
                    if predict_cls == 6:
                        forest_true_num += 1
                elif true_cls == 7:
                    playground_cell_total += 1
                    if predict_cls == 7:
                        playground_true_num += 1
                elif true_cls == 8:
                    water_cell_total += 1
                    if predict_cls == 8:
                        water_true_num += 1
                elif true_cls == 9:
                    village_cell_total += 1
                    if predict_cls == 9:
                        village_true_num +=1
                elif true_cls == 10:
                    service_cell_total += 1
                    if predict_cls == 10:
                        service_true_num += 1
                elif true_cls == 12:
                    others_cell_total += 1
                    if predict_cls == 12:
                        others_true_num += 1
                elif true_cls == 11:
                    farmland_cell_total += 1
                    if predict_cls == 11:
                        farmland_true_num += 1

    if background_cell_total == 0:
        print ("background total number is 0")
    else:
        print ("background total number is {0}, accuracy is {1}".format(background_cell_total, float(background_true_num)/background_cell_total))
    if road_cell_total == 0:
        print ("road total number is 0")
    else:
        print ("road total number is {0}, accuracy is {1}".format(road_cell_total, float(road_true_num)/road_cell_total))
    if residence_cell_total == 0:
        print ("residence total number is 0")
    else:
        print ("residence total number is {0}, accuracy is {1}".format(residence_cell_total, float(residence_true_num)/residence_cell_total))
    if industry_cell_total == 0:
        print ("industry total number is 0")
    else:
        print ("industry total number is {0}, accuracy is {1}".format(industry_cell_total, float(industry_true_num)/industry_cell_total))
    if greenland_cell_total == 0:
        print ("greenland total number is 0")
    else:
        print ("greenland total number is {0}, accuracy is {1}".format(greenland_cell_total, float(greenland_true_num)/greenland_cell_total))
    if uncompleteland_cell_total == 0:
        print ("uncompleteland total is 0")
    else:
        print ("uncompleteland total number is {0}, accuracy is {1}".format(uncompleteland_cell_total, float(uncompleteland_true_num)/uncompleteland_cell_total))
    if forest_cell_total == 0:
        print ("forest total number is 0")
    else:
        print ("forest total number is {0}, accuracy is {1}".format(uncompleteland_cell_total, float(forest_true_num)/forest_cell_total))
    if playground_cell_total == 0:
        print ("playground total is 0")
    else:
        print ("playground total number is {0}, accuracy is {1}".format(playground_cell_total, float(playground_true_num)/playground_cell_total))
    if water_cell_total == 0:
        print ("water total is 0")
    else:
        print ("water total number is {0}, accuracy is {1}".format(water_cell_total, float(water_true_num) / water_cell_total))
    if village_cell_total == 0:
        print ("village total is 0")
    else:
        print ("village total number is {0}, accuracy is {1}".format(water_cell_total, float(village_true_num) / village_cell_total))
    if service_cell_total == 0:
        print ("service total is 0")
    else:
        print ("service total number is {0}, accuracy is {1}".format(service_cell_total, float(service_true_num) / service_cell_total))
    if farmland_cell_total == 0:
        print ("farmland total is 0")
    else:
        print ("farmland total number is {0}, accuracy is {1}".format(farmland_cell_total, float(farmland_true_num) / farmland_cell_total))
    if others_cell_total == 0:
        print ("others total is 0")
    else:
        print ("others total number is {0}, accuracy is {1}".format(others_cell_total, float(others_true_num) / others_cell_total))


def calculate_total_accuracy(predict_labels,true_labels):
    total = 0
    true_num = 0
    for i, label in enumerate(predict_labels):
        for j, row in enumerate(label):
            for k, cell in enumerate(row):
                total += 1
                if cell >= 0.5:
                    predict_cls = 1
                else:
                    predict_cls = 0
                if true_labels[i][j][k] >= 0.5:
                    true_cls = 1
                else:
                    true_cls = 0
                # predict_cls = cell.argmax()
                # true_cls = true_labels[i][j][k].argmax()
                if predict_cls == true_cls:
                    true_num += 1
    total_accuracy = float(true_num) / total
    print("total accuracy is {0}".format(total_accuracy))


def calculate_error_obfuscation(predict_labels, true_labels, save_path):
    work_book = xlwt.Workbook()
    sheet = work_book.add_sheet("sheet1")
    sheet.write(0, 1, "background")
    sheet.write(0, 2, "residential_area")
    sheet.write(0, 3, "industry_area")
    sheet.write(0, 4, "server_area")
    sheet.write(0, 5, "village_area")
    sheet.write(0, 6, "forest_area")
    sheet.write(0, 7, "farmland_area")
    sheet.write(0, 8, "uncompleted_area")
    sheet.write(0, 9, "mainroad")
    sheet.write(0, 10, "all")
    sheet.write(0, 11, "accuracy")
    sheet.write(1, 0, "background")
    sheet.write(2, 0, "residential_area")
    sheet.write(3, 0, "industry_area")
    sheet.write(4, 0, "server_area")
    sheet.write(5, 0, "village_area")
    sheet.write(6, 0, "forest_area")
    sheet.write(7, 0, "farmland_area")
    sheet.write(8, 0, "uncompleted_area")
    sheet.write(9, 0, "mainroad")
    sheet.write(10, 0, "all")
    sheet.write(11, 0, "accuracy")
    for x in range(9):
        cls_x_error = calculate_error(in_predict_cls=x, predict_labels=predict_labels, true_labels=true_labels)
        for y, cell in enumerate(cls_x_error):
            sheet.write(x+1, y+1, cell)
    work_book.save(save_path)


def draw_loss(loss_list):
    ep = len(loss_list)
    x = np.linspace(1, ep, ep)
    y = loss_list
    plt.plot(x, y, 'r', linewidth=2)
    plt.xlabel(r'$\rm{epoch}  \  t$', fontsize=16)
    plt.ylabel(r'$\rm{loss} \ f(x)$', fontsize=16)
    # plt.title(r'$f(x) \ \rm{is \ damping  \ with} \ x$', fontsize=16)
    # plt.text(2.0, 0.5, r'$f(x) = \rm{sin}(2 \pi  x^2) e^{\sigma x}$', fontsize=20)
    plt.savefig('bg_val_loss.png', dpi=75)
    plt.show()


def calculate_error(in_predict_cls, predict_labels,true_labels):
    total = 0
    true_num = 0
    background_num = 0
    residential_area_num = 0
    industry_area_num = 0
    server_area_num = 0
    village_area_num= 0
    forest_area_num = 0
    farmland_area_num = 0
    uncompleted_area_num = 0
    mainroad_num = 0
    for i, label in enumerate(predict_labels):
        for j, row in enumerate(label):
            for k, cell in enumerate(row):
                predict_cls = cell.argmax()
                true_cls = true_labels[i][j][k].argmax()
                if true_cls == in_predict_cls:
                    total += 1
                    if predict_cls == 0:
                        background_num += 1
                        if true_cls == predict_cls:
                            true_num += 1
                    elif predict_cls == 1:
                        residential_area_num += 1
                        if true_cls == predict_cls:
                            true_num += 1
                    elif predict_cls == 2:
                        industry_area_num += 1
                        if true_cls == predict_cls:
                            true_num += 1
                    elif predict_cls == 3:
                        server_area_num += 1
                        if true_cls == predict_cls:
                            true_num += 1
                    elif predict_cls == 4:
                        village_area_num += 1
                        if true_cls == predict_cls:
                            true_num += 1
                    elif predict_cls == 5:
                        forest_area_num += 1
                        if true_cls == predict_cls:
                            true_num += 1
                    elif predict_cls == 6:
                        farmland_area_num += 1
                        if true_cls == predict_cls:
                            true_num += 1
                    elif predict_cls == 7:
                        uncompleted_area_num += 1
                        if true_cls == predict_cls:
                            true_num += 1
                    elif predict_cls == 8:
                        mainroad_num +=1
                        if true_cls == predict_cls:
                            true_num += 1
                   
    if total != 0:
        in_cls_accuracy = float(true_num) / total
    else:
        in_cls_accuracy = 0.0
    return [background_num, residential_area_num, industry_area_num, server_area_num, village_area_num,
                forest_area_num, farmland_area_num, uncompleted_area_num, mainroad_num, total, in_cls_accuracy]


if __name__ == "__main__":
    # with open("C:/Users/29625\Desktop/sm_val_loss.txt",'r') as f:
    #     lines = f.readlines()
    # loss_list = []
    # for line in lines:
    #     loss_list.append(float(line.strip("\n")))
    # draw_loss(loss_list)
    # x= float(1e-5)
    # print(x)
    # convert_to_imgs(npy_path=r"Z:\xin.data\data\mlw\data", save_path=r"Z:\xin.data\data\mlw\data\res")
    #create_tes_label("data/big_scale/test_label", "data/big_scale/npydata", 200, 200, "tif")
    # predict_labels = np.load("../../sources/unet/npydata/imgs_mask_predict.npy")
    # true_labels = np.load("../../sources/unet/npydata/train_label_np.npy")
    # predict_labels = np.load(r"G:\xin.data\rs\mlw\data/imgs_mask_predict.npy")
    # true_labels = np.load(r"G:\xin.data\rs\mlw\data/valid_label_np.npy")
    # # # # # calculate_accuracy(predict_labels, true_labels)
    # calculate_total_accuracy(predict_labels, true_labels)
    # calculate_error_obfuscation(predict_labels=predict_labels, true_labels=true_labels, save_path="train_error.xls")
    # print("begin to class sample.....")
    # class_sample()
    # print("finished class sample.....")
    # split_sample(r'G:\xin.data\rs\wf\gt', r'G:\xin.data\rs\wf\resize_image',
    #              r'G:\xin.data\rs\wf\data\train_image', r'G:\xin.data\rs\wf\data\train_label',
    #              r'G:\xin.data\rs\wf\data\val_image', r'G:\xin.data\rs\wf\data\val_label')
    create_train_data(train_img_path=r"G:\xin.data\datasets\nb_rs\part_data\train\img",
                      train_label_path=r"G:\xin.data\datasets\nb_rs\part_data\train\gt",
                      npy_path=r"G:\xin.data\datasets\nb_rs\part_data", row_num=256, column_num=256, file_type="png")
    create_train_data(train_img_path=r"G:\xin.data\datasets\nb_rs\part_data\val\img",
                      train_label_path=r"G:\xin.data\datasets\nb_rs\part_data\val\gt",
                      npy_path=r"G:\xin.data\datasets\nb_rs\part_data", row_num=256, column_num=256, file_type="png",
                      is_valid_data=True)
    create_tes_data(test_img_path=r"G:\xin.data\datasets\nb_rs\part_data\val\img",
                    npy_path=r"G:\xin.data\datasets\nb_rs\part_data", row_num=256,
                     column_num=256, file_type="png")
    # x = np.linspace(1,60,60)
    # print(x)

