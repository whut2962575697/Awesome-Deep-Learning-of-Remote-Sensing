# Awesome Deep Learning of Remote Sensing [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
In this project, we will open source some baseline codes for the remote sensing analysis task, such as semantic segmentation, scene classification, object detection, and image captioning, We will also collect some public datasets that can be used for remote sensing image research and analysis.
- [x] Public Remote Sensing Dataset
- [x] Baseline Codes (Semantic Segmentation/Scene Classification/Object Detection/Image Captioning)
- [x] OpenSoure Codes
- [x] Compitions About Remote Sensing

## Public Remote Sensing Dataset

### 1.Semantic Segmentation
##### [ISPRS Potsdam 2D Semantic Labeling Contest ](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html "ISPRS Potsdam 2D Semantic Labeling Contest")   
6 urban land cover classes, raster mask labels, 4-band RGB-IR aerial imagery (0.05m res.) & DSM, 38 image patches   
##### categories      
 | index | label | color |
 | :-----| ----: | :----: |
 | 1 | Impervious surfaces | 255, 255, 255 |
 | 2 | Building | 0, 0, 255 |
 | 3 | Low vegetation | 0, 255, 255 |
 | 4 | Tree | 0, 255, 0 |
 | 5 | Car | 255, 255, 0 |
 | 6 | Clutter/background | 255, 0, 0 | 
##### Download   
[baiduyun password: 9enz](https://pan.baidu.com/s/1l_s8XsT_wn5TgpNqwMnVMw "ISPRS Potsdam 2D Semantic Labeling Contest")   
#### [DSTL Satellite Imagery Feature Detection Challenge ](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection "DSTL Satellite Imagery Feature Detection Challenge ")   
10 land cover categories from crops to vehicle small, 57 1x1km images, 3/16-band Worldview 3 imagery (0.3m-7.5m res.), Kaggle kernels
#### [Slovenia Land Cover Classification](http://eo-learn.sentinel-hub.com/ "Slovenia Land Cover Classification")   
10 land cover classes, temporal stack of hyperspectral Sentinel-2 imagery (R,G,B,NIR,SWIR1,SWIR2; 10 m res.) for year 2017 with cloud masks, Official Slovenian land use land cover layer as ground truth.
#### [SEN12MS](https://mediatum.ub.tum.de/1474000 "SEN12MS")    
180,748 corresponding image triplets containing Sentinel-1 (VV&VH), Sentinel-2 (all bands, cloud-free), and MODIS-derived land cover maps (IGBP, LCCS, 17 classes, 500m res.). All data upsampled to 10m res., georeferenced, covering all continents and meterological seasons, Paper: Schmitt et al. 2018
#### [IEEE Data Fusion Contest 2018 ](http://www.grss-ieee.org/community/technical-committees/data-fusion/2018-ieee-grss-data-fusion-contest/ "IEEE Data Fusion Contest 2018 ")   
20 land cover categories by fusing three data sources: Multispectral LiDAR, Hyperspectral (1m), RGB imagery (0.05m res.)
#### [Segmentation Data of Sparse Representation and Intelligent Analysis of 2019 Remote Sensing Image competition]( "") (website has been closed)   
16 land cover classes,4-band RGB-IR aerial imagery (4m res.) 8 patches of 7200x6800 for train and 2 patches of 7200x6800 for val and 10 patches of 7200x6800 for test
##### categories  
 | index | label | color |
 | :-----| ----: | :----: |
 | 1 | 水田 | 0,200,0 |
 | 2 | 水浇田 | 150,250,0 |
 | 3 | 旱耕地 | 150,200,150 |
 | 4 | 园地 | 200,0,200 |
 | 5 | 乔木林地 | 150,0,250 |
 | 6 | 灌木林地 | 150,150,250 | 
 | 7 | 天然草地 | 250,200,0 | 
 | 8 | 人工草地 | 200,200,0 | 
 | 9 | 工业用地 | 200,0,0 |
 | 10 | 城市住宅 | 250,0,150 |
 | 11 | 村镇住宅 | 200,150,150 |
 | 12 | 交通运输 | 250,150,150 |
 | 13 | 河流 | 0,0,200 |
 | 14 | 湖泊 | 0,150,200 |
 | 15 | 坑塘 | 0,200,250 |
 | 16 | 其他 | 0,0,0 |
 ##### Download 
 [baiduyun password: o2fp](https://pan.baidu.com/s/1Bun1xlFFr49HXJXSjKab3Q "Segmentation Data of Sparse Representation and Intelligent Analysis of 2019 Remote Sensing Image competition")
#### [2019 年县域农业大脑AI挑战赛](https://tianchi.aliyun.com/competition/entrance/231717/information "2019 年县域农业大脑AI挑战赛")   
5 argriculture categories
#### [CCF 卫星影像的AI分类与识别比赛 BDCI 2017](https://www.datafountain.cn/competitions/270/datasets "BDCI 2017")   
5 land cover classes(greenland, building, waterbody, road and other), 5 rgb images(R,G,B; 1 m res.) for train and val, 3 rgb images for test
##### categories
 | index | label | gary |
 | :-----| ----: | :----: |
 | 1 | 植被 | 1 |
 | 2 | 建筑 | 2 |
 | 3 | 水体 | 3 |
 | 4 | 道路 | 4 |
 | 5 | 其他 | 0 | 
 ##### Download 
 [baiduyun password: aaod](https://pan.baidu.com/s/1ld8p2bghiKu_2A6Mg4_-JQ "BDCI 2017")
#### [2020 NAIC “华为・昇腾杯”AI+遥感影像 ](https://naic.pcl.ac.cn/frame/2 "2020NAIC")   
- 初赛：10万高分光学影像和标注文件（一级分类（8类）），20万测试图片数据；
- 复赛：10万高分光学影像和标注文件（二级分类（17类）），30万测试图片数据；
##### 初赛 categories  
 | index | 一级标签 | gary(百位数字) |
 | :-----| ----: | :----: |
 | 1 | 水体 | 1 |
 | 2 | 交通运输 | 2 |
 | 3 | 建筑 | 3 |
 | 4 | 耕地 | 4 |
 | 5 | 草地 | 5 |
 | 6 | 林地 | 6 | 
 | 7 | 裸土 | 7 | 
 | 8 | 其他 | 8 |    
 ##### 复赛 categories  
 | index | 二级标签 | gary(十位及个位上的数字) |
 | :-----| ----: | :----: |
 | 1 | 水体 | 01 |
 | 2 | 道路 | 02 |
 | 3 | 建筑物 | 03 |
 | 4 | 机场 | 04 |
 | 5 | 火车站 | 05 |
 | 6 | 光伏 | 06 | 
 | 7 | 停车场 | 07 | 
 | 8 | 操场 | 08 | 
 | 9 | 普通耕地 | 09 |
 | 10 | 农业大棚 | 10 |
 | 11 | 自然草地 | 11 |
 | 12 | 绿地绿化 | 12 |
 | 13 | 自然林 | 13 |
 | 14 | 人工林 | 14 |
 | 15 | 自然裸土 | 15 |
 | 16 | 人为裸土 | 16 |
 | 17 | 其他 | 17 |
 ##### Download   
[baiduyun password: pdos](https://pan.baidu.com/s/1Q4YjUAhmTwDCQA7wlKrDxQ "NAIC RS 2020")    
#### [2020 CCF BDCI 遥感影像地块分割 ](https://www.datafountain.cn/competitions/475/datasets "2020BDCI")   
训练集包含140,000张分辨率为2m/pixel，尺寸为256256的JPG图片，一共7个类别，对应gt 0-6
##### categories
 | index | label | gary |
 | :-----| ----: | :----: |
 | 1 | 建筑 | 0 |
 | 2 | 耕地 | 1 |
 | 3 | 林地 | 2 |
 | 4 | 水体 | 3 |
 | 5 | 道路 | 4 |
 | 6 | 草地 | 5 | 
 | 7 | 其他 | 6 | 
 | 8 | 未标注区域 | 255 | 
 ##### Download 
 [baiduyun password: 7tcn](https://pan.baidu.com/s/1bG9SdvXzmp3oNjkTPkwwfw "BDCI 2020")  
 
 #### [2021全国数字生态创新大赛智能算法赛：生态资产智能分析 ](https://tianchi.aliyun.com/competition/entrance/531860/introduction "2021TCRS")   
初赛训练集包含16017张分辨率为0.8m-2m/pixel，尺寸为256256的TIF图片，一共10个类别，对应gt 1-10   
复赛训练集包含15904张分辨率为0.8m-2m/pixel，尺寸为256256的TIF图片，一共10个类别，对应gt 1-10
##### categories
 | index | label | gary |
 | :-----| ----: | :----: |
 | 1 | 耕地 | 1 |
 | 2 | 林地 | 2 |
 | 3 | 草地 | 3 |
 | 4 | 道路 | 4 |
 | 5 | 城镇建设用地 | 5 |
 | 6 | 农村建设用地 | 6 | 
 | 7 | 工业用地 | 7 | 
 | 8 | 构筑物 | 8 | 
 | 9 | 水域 | 9 | 
 | 10 | 裸地 | 10 | 
 ##### Download 
 [baiduyun password: td5k](https://pan.baidu.com/s/1QQJPTKB8zqEflYmHaTXtJg "2021TCRS") 
 

### 2.Scene Classification
### 3.Object Detection
### 4.Image Captioning
## Baseline Codes
## OpenSoure Codes
### 1.Semantic Segmentation
- [cuilunan/Unet-of-remote-sensing-image](https://github.com/cuilunan/Unet-of-remote-sensing-image "cuilunan") [tensorflow]
- [Epsilon123/Semantic-Segmentation-of-Remote-Sensing-Images](https://github.com/Epsilon123/Semantic-Segmentation-of-Remote-Sensing-Images "Epsilon123") [keras]
- [YudeWang/UNet-Satellite-Image-Segmentation](https://github.com/YudeWang/UNet-Satellite-Image-Segmentation "YudeWang") [tensorflow]
- [rmkemker/EarthMapper](https://github.com/rmkemker/EarthMapper "rmkemker/EarthMapper") [tensorflow]
- [TachibanaYoshino/Remote-sensing-image-semantic-segmentation](https://github.com/TachibanaYoshino/Remote-sensing-image-semantic-segmentation "TachibanaYoshino/Remote-sensing-image-semantic-segmentation") [Keras]
- [lcylmhlcy/Semantic-segmentation](https://github.com/lcylmhlcy/Semantic-segmentation "lcylmhlcy/Semantic-segmentation") [pytorch]
- [liushuo2018/ERN](https://github.com/liushuo2018/ERN "liushuo2018/ERN") [caffe]
- [Walkerlikesfish/HSNRS](https://github.com/Walkerlikesfish/HSNRS "Walkerlikesfish/HSNRS") [caffe]
- [1044197988/Semantic-segmentation-of-remote-sensing-images](https://github.com/1044197988/Semantic-segmentation-of-remote-sensing-images "1044197988/Semantic-segmentation-of-remote-sensing-images") [keras]
- [fuweifu-vtoo/Semantic-segmentation](https://github.com/fuweifu-vtoo/Semantic-segmentation "fuweifu-vtoo/Semantic-segmentation") [pytorch]
- [reachsumit/deep-unet-for-satellite-image-segmentation](https://github.com/reachsumit/deep-unet-for-satellite-image-segmentation "reachsumit/deep-unet-for-satellite-image-segmentation") [keras]
- [lehaifeng/SCAttNet](https://github.com/lehaifeng/SCAttNet "lehaifeng/SCAttNet") [tensorflow]
- [NexGenMap/dl-semantic-segmentation](https://github.com/NexGenMap/dl-semantic-segmentation "NexGenMap/dl-semantic-segmentation") [tensorflow ]
- [yiskw713/boundary_loss_for_remote_sensing](https://github.com/yiskw713/boundary_loss_for_remote_sensing "yiskw713/boundary_loss_for_remote_sensing") [pytorch]
- [zetrun-liu/FCNs-for-road-extraction-keras](https://github.com/zetrun-liu/FCNs-for-road-extraction-keras "zetrun-liu/FCNs-for-road-extraction-keras") [keras]
- [susurrant/rs-img-classification](https://github.com/susurrant/rs-img-classification "susurrant/rs-img-classification") [tensorflow]
- [AI-Chen/Deeplab-v3-Plus-pytorch-](https://github.com/AI-Chen/Deeplab-v3-Plus-pytorch- "AI-Chen/Deeplab-v3-Plus-pytorch-") [pytorch]
- [mohuazheliu/ResUnet-a](https://github.com/mohuazheliu/ResUnet-a "mohuazheliu/ResUnet-a") [tensorflow]
- [zlkanata/DeepGlobe-Road-Extraction-Challenge](https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge "zlkanata/DeepGlobe-Road-Extraction-Challenge") [pytorch]
- [DeepVoltaire/Dstl-Satellite-Imagery-Feature-Detection](https://github.com/DeepVoltaire/Dstl-Satellite-Imagery-Feature-Detection "DeepVoltaire/Dstl-Satellite-Imagery-Feature-Detection") [keras]
### 2.Scene Classification
- [weihancug/SENet_ResNeXt_Remote_Sensing_Scene_Classification](https://github.com/weihancug/SENet_ResNeXt_Remote_Sensing_Scene_Classification "weihancug/SENet_ResNeXt_Remote_Sensing_Scene_Classification") [pytorch]
- [BiQiWHU/DenseNet40-for-HRRSISC](https://github.com/BiQiWHU/DenseNet40-for-HRRSISC "BiQiWHU/DenseNet40-for-HRRSISC") [tensorflow]
- [weihancug/SSGF-for-HRRS-scene-classification](https://github.com/weihancug/SSGF-for-HRRS-scene-classification "weihancug/SSGF-for-HRRS-scene-classification") [caffe]
- [Arafat123-iit/A-System-for-Effecient-Remote-Sensing-Image-Scene-Classification-](https://github.com/Arafat123-iit/A-System-for-Effecient-Remote-Sensing-Image-Scene-Classification- "Arafat123-iit/A-System-for-Effecient-Remote-Sensing-Image-Scene-Classification-") [keras]
- [Aaromxj/SF-CNN](https://github.com/Aaromxj/SF-CNN "Aaromxj/SF-CNN") [Caffe]
- [Aaron-Lst/ARCNet](https://github.com/Aaron-Lst/ARCNet "Aaron-Lst/ARCNet") [pytorch]
- [Wanke15/Feature_extraction-SVM-classification-Remote-sensing](https://github.com/Wanke15/Feature_extraction-SVM-classification-Remote-sensing "Wanke15/Feature_extraction-SVM-classification-Remote-sensing") [caffe]
- [williamzhao95/Pay-More-Attention](https://github.com/williamzhao95/Pay-More-Attention "williamzhao95/Pay-More-Attention") [Mxnet]
- [henanjun/SccovNet](https://github.com/henanjun/SccovNet "henanjun/SccovNet") [matlab]
### 3.Object Detection
- [clw5180/remote_sensing_object_detection_2019](https://github.com/clw5180/remote_sensing_object_detection_2019 "clw5180/remote_sensing_object_detection_2019") [pytorch]
- [jiangruoqiao/RICNN_GongCheng_CVPR2015](https://github.com/jiangruoqiao/RICNN_GongCheng_CVPR2015 "jiangruoqiao/RICNN_GongCheng_CVPR2015") [tensorflow]
- [R-Stefano/Remote-Sensing-Analysis](https://github.com/R-Stefano/Remote-Sensing-Analysis "R-Stefano/Remote-Sensing-Analysis") [tensorflow]
- [WenchaoliuMUC/Detection-of-Multiclass-Objects-in-Optical-Remote-Sensing-Images](https://github.com/WenchaoliuMUC/Detection-of-Multiclass-Objects-in-Optical-Remote-Sensing-Images "WenchaoliuMUC/Detection-of-Multiclass-Objects-in-Optical-Remote-Sensing-Images") [pytorch]
- [weihancug/Remote-Sensing-Object-Detection-with-Oriented-Bouding-Box](https://github.com/weihancug/Remote-Sensing-Object-Detection-with-Oriented-Bouding-Box "weihancug/Remote-Sensing-Object-Detection-with-Oriented-Bouding-Box") [pytorch]
- [Pilot-Zhang/ssd.pytorch](https://github.com/Pilot-Zhang/ssd.pytorch "Pilot-Zhang/ssd.pytorch") [pytorch]
### 4.Image Captioning
## Compitions About Remote Sensing
### 2020
### 2019
### 2018
### 2017
