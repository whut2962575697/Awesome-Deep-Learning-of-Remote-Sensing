# -*- encoding: utf-8 -*-
'''
@File    :   fcn8.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/4/20 16:20   xin      1.0         None
'''

from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, \
    Dense, BatchNormalization,Flatten, Reshape, Activation, Cropping2D, Add
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import numpy as np

from data_batch import imageSegmentationGenerator

VGG_Weights_path = "data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


class Tool(object):
    def __init__(self, npy_path):
        self.npy_path = npy_path

    def load_train_data(self, is_valid_data=False):
        if is_valid_data:
            imgs = np.load(self.npy_path + "valid_img_np.npy")
            imgs = imgs.astype("float32")
            imgs /= 255
            labels = np.load(self.npy_path + "valid_label_np.npy")
            return imgs, labels
        else:
            imgs = np.load(self.npy_path + "train_img_np.npy")
            imgs = imgs.astype("float32")
            imgs /= 255
            labels = np.load(self.npy_path + "train_label_np.npy")
            return imgs, labels

    def load_tes_data(self):
        imgs = np.load(self.npy_path + "test_img_np.npy")
        imgs = imgs.astype("float32")
        imgs /= 255
        return imgs


class FCN8(object):
    def __init__(self, img_rows=224, img_columns=224, n_cls=13):
        self.img_rows = img_rows
        self.img_columns = img_columns
        self.n_cls = n_cls

    def load_data(self):
        tool = Tool(r"G:\xin.data\rs\mlw\fcn\FCN_sample/")
        train_imgs, train_labels = tool.load_train_data()
        valid_imgs, valid_labels = tool.load_train_data(is_valid_data=True)
        test_imgs = tool.load_tes_data()
        return train_imgs, train_labels, test_imgs, valid_imgs, valid_labels

    def crop(self, o1, o2, i):
        o_shape2 = Model(i, o2).output_shape
        outputHeight2 = o_shape2[1]
        outputWidth2 = o_shape2[2]

        o_shape1 = Model(i, o1).output_shape
        outputHeight1 = o_shape1[1]
        outputWidth1 = o_shape1[2]

        cx = abs(outputWidth1 - outputWidth2)
        cy = abs(outputHeight2 - outputHeight1)

        if outputWidth1 > outputWidth2:
            o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

        if outputHeight1 > outputHeight2:
            o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

        return o1, o2

    def get_model(self):
        inputs = Input((self.img_rows, self.img_columns, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # conv1 = BatchNormalization()(conv1)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        # conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

        print("conv2 shape:", conv2.shape)
        # conv2 = BatchNormalization()(conv2)
        # bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        # conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        # conv3 = BatchNormalization()(conv3)
        # bn3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        # conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # conv4 = BatchNormalization()(conv4)
        print("drop4 shape:", conv4.shape)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        print("pool4 shape:", pool4.shape)

        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        # conv4 = BatchNormalization()(conv4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        # conv4 = BatchNormalization()(conv4)
        print("drop5 shape:", conv5.shape)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
        print("pool5 shape:", pool5.shape)

        # flatten = Flatten(name='flatten')(pool5)
        # fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
        # fc2 = Dense(4096, activation='relu', name='fc2')(fc1)
        # fc3 = Dense(1000, activation='softmax', name='predictions')(fc2)

        vgg = Model(inputs, pool5)
        vgg.load_weights(VGG_Weights_path)

        up_conv1 = Conv2D(4096, 7, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
        # conv4 = BatchNormalization()(conv4)
        dropout1 = Dropout(0.5)(up_conv1)
        up_conv1 = Conv2D(4096, 1, activation='relu', padding='same', kernel_initializer='he_normal')(up_conv1)
        # conv4 = BatchNormalization()(conv4)
        print("up_conv1 shape:", up_conv1.shape)
        dropout1 = Dropout(0.5)(up_conv1)

        up_conv2 = Conv2D(self.n_cls, 7, activation='relu', padding='same', kernel_initializer='he_normal')(dropout1)
        # conv4 = BatchNormalization()(conv4)
        up_conv2 = Conv2DTranspose(self.n_cls, 4, strides=(2, 2), use_bias=False)(up_conv2)
        # conv4 = BatchNormalization()(conv4)
        print("up_conv2 shape:", up_conv2.shape)

        crop1 = (Conv2D(self.n_cls, 1, kernel_initializer='he_normal'))(pool4)

        up_conv2, crop1 = self.crop(up_conv2, crop1, inputs)

        merge1 = Add()([up_conv2, crop1])
        print("merge1 shape:", merge1.shape)
        up_conv3 = Conv2DTranspose(self.n_cls, 4, strides=(2, 2), use_bias=False)(merge1)
        print("up_conv3 shape:", up_conv3.shape)
        crop2 = (Conv2D(self.n_cls, 1, kernel_initializer='he_normal'))(pool3)
        up_conv3, crop2 = self.crop(up_conv3, crop2, inputs)
        merge2 = Add()([up_conv3, crop2])
        print("merge2 shape:", merge2.shape)
        up_conv4 = Conv2DTranspose(self.n_cls, 40, strides=(8, 8), use_bias=False)(merge2)
        print("up_conv4 shape:", up_conv4.shape)


        o_shape = Model(inputs, up_conv4).output_shape

        outputHeight = o_shape[1]
        outputWidth = o_shape[2]
        print("output_shape", o_shape)

        # o = (Reshape((-1, outputHeight * outputWidth, self.n_cls)))(up_conv2)
        o = (Activation('softmax'))(up_conv4)
        model = Model(inputs, o)
        model.outputWidth = outputWidth
        model.outputHeight = outputHeight

        model.compile(optimizer=Adam(lr=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, exist_model_path=None):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test, imgs_valid, imgs_mask_valid = self.load_data()
        print("loading data done")
        if exist_model_path:
            model = load_model(exist_model_path)
        else:
            model = self.get_model()

        print("got fcn8")

        model_checkpoint = ModelCheckpoint('fcn8.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        history = model.fit(imgs_train, imgs_mask_train, batch_size=1, nb_epoch=50, verbose=1, validation_split=0.2,
                            validation_data=(imgs_valid, imgs_mask_valid), shuffle=True, callbacks=[model_checkpoint])
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        np_loss = np.array(loss)
        np_val_loss = np.array(val_loss)
        np.savetxt("loss.txt", np_loss)
        np.savetxt("val_loss.txt", np_val_loss)
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('imgs_mask_predict.npy', imgs_mask_test)

    def train_batch(self, train_images_path, train_segs_path, val_images_path, val_segs_path, exist_model_path=None):
        if exist_model_path:
            model = load_model(exist_model_path)
        else:
            model = self.get_model()
        model_checkpoint = ModelCheckpoint('fcn8.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')

        G = imageSegmentationGenerator(train_images_path, train_segs_path, 1, 13,
                                                   224, 224, 256, 256)

        G2 = imageSegmentationGenerator(val_images_path, val_segs_path, 1, 13,
                                       224, 224, 256, 256)

        history = model.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1, callbacks=[model_checkpoint])
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        np_loss = np.array(loss)
        np_val_loss = np.array(val_loss)
        np.savetxt("loss.txt", np_loss)
        np.savetxt("val_loss.txt", np_val_loss)
        print('predict test data')


if __name__ == "__main__":
    fcn8 = FCN8()
    fcn8.train()
