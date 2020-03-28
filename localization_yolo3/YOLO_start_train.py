import os
import random
import xml.etree.ElementTree as ET
from os import getcwd
import train_yolov3;
import shutil
""""
样本检查函数

"""
def wrong_file_del(path):
    wrong_file = [];
    with open(path) as f_train:
        lines = f_train.readlines();
        for train_path in lines:
            if 'jpg' in train_path[-6:]:
                wrong_file.append(train_path);

    ret = [new_file for new_file in lines if new_file not in wrong_file];
    data_length = len(ret);
    f_new = open(path, 'w');
    ret = "".join(ret);
    if len(ret) == 0:
        exit("2007_train.txt文件不满足要求,程序停止运行,请检查文件拷贝过程");
    else:
        print("文件个数为:",data_length)
        f_new.write(ret);
        f_new.close();
        print("文件检查完成.........");

""""

    设置标注的路径和图片路径

"""
path_Annotations_original = "E:/yolo/VOCdevkit/VOC2007/Annotations/";
path_JPEGImages_original = "E:/yolo/VOCdevkit/VOC2007/JPEGImages/";

yolo_anchor = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326";

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2007', 'trainval')]

""""

    类别+文件存放路径(类别可修改)

"""
classes = ["Arc"]
cwd = getcwd();
wd  = cwd;
VOCdevkit_path = cwd+'/VOCdevkit/';
os.mkdir(VOCdevkit_path) if not os.path.isdir(VOCdevkit_path) else None;
Annotations_path = cwd+'/VOCdevkit/VOC2007/Annotations/';
os.makedirs(Annotations_path) if not os.path.isdir(Annotations_path) else None;
JPEGImages_path = cwd+'/VOCdevkit/VOC2007/JPEGImages/';
os.mkdir(JPEGImages_path) if not os.path.isdir(JPEGImages_path) else None;
ImageSets_path = cwd+'/VOCdevkit/VOC2007/ImageSets/';
os.mkdir(ImageSets_path) if not os.path.isdir(ImageSets_path) else None;
mode_data_path = cwd+"/model_data/";
os.mkdir(mode_data_path) if not os.path.isdir(mode_data_path) else None;
main_path = cwd+'/VOCdevkit/VOC2007/ImageSets/Main/';
os.mkdir(main_path) if not os.path.isdir(main_path) else None;


""""

    文件拷贝

"""
print("*************************************************");
orignal_Annotations = int(input("标注框拷贝到VOCdevkit/VOC2007/Annotations/路径下,(1-已经拷贝,0-未拷贝):"));
print("*************************************************");
orignal_JPEGImage = int(input("图片拷贝到VOCdevkit/VOC2007/JPEGImages/路径下,(1-已经拷贝,0-未拷贝):"));
print("*************************************************");
if not orignal_Annotations:
    for Annotation in os.listdir(path_Annotations_original):
        shutil.copy(path_Annotations_original+Annotation, Annotations_path);

if not orignal_JPEGImage:
    for JPEGImage in os.listdir(path_JPEGImages_original):
        shutil.copy(path_JPEGImages_original+JPEGImage, JPEGImages_path);

    """"

    设置训练，验证，测试样本
    写入类别

    """
trainval_percent = 0.03;
train_percent = 0.97;
total_xml = os.listdir(Annotations_path);
num = len(total_xml);
list = range(num);
tv = int(num * trainval_percent);
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
orignal_classes = int(input("类别和anchors发生改变,(1-发生,0-未发生):"));
print("*************************************************\n");
if orignal_classes:
    name_change= input("请输入改变的类名:");
    classes[0] = name_change;
    fwrite_coco = open(cwd+'/model_data/coco_classes.txt', 'w');
    fwrite_voc = open(cwd+'/model_data/voc_classes.txt', 'w');
    fwrite_anchor = open(cwd+'/model_data/yolo_anchors.txt', 'w');
    for name in classes:
        fwrite_coco.write(name +'\n');
        fwrite_voc.write(name+'\n');
    fwrite_anchor.write(yolo_anchor);
    fwrite_coco.close();
    fwrite_voc.close();
    fwrite_anchor.close();


#写入样本
ftrainval = open(cwd+'/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')
ftest = open(cwd+'/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
ftrain = open(cwd+'/VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')
fval = open(cwd+'/VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

#重写类别


fwrite_voc = open(cwd+'/model_data/voc_classes.txt', 'r');
classes[0] = ((fwrite_voc.readline()).split("\n"))[0];
fwrite_voc.close();
print("检测到的类别为: %s" %classes);

#读燃弧区域坐标
def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))



#转换成yolo v3训练模型文件格式，写入2007_train.txt文件
for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()



#样本检查
print("正在检查2007_train.txt 文件是否 (全部) 满足格式......");
jugle = wrong_file_del(cwd+'/2007_train.txt');

print("-------------------------------------------------------------\n");
string = int(input("检查完成,数字 1 开始YOLO-V3 模型训练，(1-开始训练,0-退出):"));




#模型训练
if string == 1:
    print("*************************************************");
    print("开始YOLO V3 模型训练...............");
    train_yolov3.train_start();
else:
    print("没有进行模型训练");