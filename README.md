# Awesome Deep Learning of Remote Sensing [图片](https://github.com/sindresorhus/awesome)
- [x] Public Remote Sensing Dataset
- [x] Baseline Code (Semantic Segmentation/Scene classification/Object Detection)
- [x] Other OpenSoure Codes
- [x] Compitions About Remote Sensing



其中fcn32.py中为fcn-32s代码，fcn8.py中为fcn-8s代码，包括模型的搭建和训练过程。


1、使用时需要制作数据集，该项目不提供遥感分割数据集，然后利用vgg_resize.py将原始影像图片resize为224*224，再利用prepare_data.py将影像图片和gt图片保存为npy矩阵文件。

2、下载vgg16预训练模型，地址为： [vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)

3、根据需要配置config.py文件，其中定义的gt中rgb值对应的one-hot向量的索引及其对应的地物类别

4、修改fcn8.py(fcn32.py)中的相关路径配置

如下：

VGG_Weights_path = "data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" # vgg16预训练模型

tool = Tool(r"G:\xin.data\rs\mlw\fcn\FCN_sample\") # npy文件保存的路径

设置合适的batch_size参数与学习率参数，batch_size建议不要设置太大，否则可能显存不够用。

5、训练时直接运行fcn8.py(fcn32.py)

6、训练结束后模型文件保存在当前路径下fcn8.hdf5

7、模型训练完毕即可调用模型进行预测，预测的代码在predict.py中

部分数据如下：

[遥感影像1]:https://github.com/whut2962575697/fcn/blob/master/data/samples/img/1068.png

[gt图片1]:https://github.com/whut2962575697/fcn/blob/master/data/samples/gt/1068.png

![遥感影像1] ![gt图片1]
 

[遥感影像2]:https://github.com/whut2962575697/fcn/blob/master/data/samples/img/1268.png

[gt图片2]:https://github.com/whut2962575697/fcn/blob/master/data/samples/gt/1268.png

![遥感影像2] ![gt图片2]
