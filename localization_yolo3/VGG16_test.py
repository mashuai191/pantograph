# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:45:09 2020

@author: CZZ
"""
import glob
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os;
import numpy as np;
path_current = os.getcwd();
finetune_vgg16_with_img_aug=load_model(path_current+'\\'+'vgg16_img_aug_weight.h5')

num2class_label_transformer = lambda l: ['ayes' if x == 0 else 'nigh' for x in l]
class2num_label_transformer = lambda l: [0 if x == 'ayes' else 1 for x in l]

IMG_DIM = (224,224)
test_files = glob.glob('test_data/*')

test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255

test_labels = train_labels = [(fn.split('\\')[1])[0:4] for fn in test_files];
test_labels_enc = class2num_label_transformer(test_labels)
print(test_labels[0:5], test_labels_enc[0:5])
import time;
time_start = time.time();
predictions_enc = finetune_vgg16_with_img_aug.predict_classes(test_imgs_scaled, verbose=0)
time_end = time.time();
print("200图片用时：%f" % float(time_end-time_start));
predictions = num2class_label_transformer(predictions_enc)

print(path_current);