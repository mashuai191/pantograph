# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

""""
注意问题：

1.每一次需要把原图（未标注的图片放入VOCdevkit\VOC2007\JPEGImages\文件夹下）；
(注意每一次标注的名字保持一致)
2.使用labelImg工具标注受电弓燃狐区域，产生XML文件，将该文件放入OCdevkit\VOC2007\Annotations\文件夹下。
3.文件读取成功与否可以看当前文件夹下2007_train.txt文件是否包含文件路径，以及训练图片燃狐区域
正确的用于训练的YOLO文件如下：
E:\keras-yolo3-master/VOCdevkit/VOC2007/JPEGImages/night10179.jpg 1083,484,1132,531,0
E:\keras-yolo3-master/VOCdevkit/VOC2007/JPEGImages/night11856.jpg 1136,395,1202,448,0 1146,461,1207,505,0
4.当前文件夹下文件夹作用
model_data :用于识别的类别,权重等文件
注意安装的软件版本：
python版本：3.6.2
TensorFlow版本：1.14.0
Keras版本：2.1.5
numpy版本：1.16.4
如果不安装对应的版本，可能会出现版本不兼容，程序无法执行
模型训练：YOLO_start.py (训练之前记得把标注文件和图片放入相应的文件夹下)
检测：yolo_video.py
标注工具：windows_v1.5.1
"""


---

