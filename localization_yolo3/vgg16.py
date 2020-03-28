import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os
from PIL import Image
from keras.models import load_model
import shutil
from keras.applications import vgg16
from keras.models import Model
import keras
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, BatchNormalization, Activation
from keras.models import Sequential
from keras import optimizers
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc
from PIL import Image


path = os.getcwd();
np.random.seed(2)
files=glob.glob('VGG16_train/*')
print(len(files))
have_files=[fn for fn in files if 'ayes' in fn]
no_files=[fn for fn in files if 'night' in fn]
print(len(have_files),len(no_files))

#选择训练样本
have_train=np.random.choice(have_files, size=len(have_files)-200, replace=False)
no_train=np.random.choice(no_files, size=len(no_files)-200, replace=False)
have_files=list(set(have_files) - set(have_train))
no_files=list(set(no_files) - set(no_train))

#验证集
have_val=np.random.choice(have_files, size=100, replace=False)
no_val=np.random.choice(no_files, size=100, replace=False)
have_files=list(set(have_files)-set(have_val))
no_files=list(set(no_files)-set(no_val))

#测试集
have_test=np.random.choice(have_files, size=100, replace=False)
no_test=np.random.choice(no_files, size=100, replace=False)
print('Have_datasets:',have_train.shape,have_val.shape,have_test.shape)
print('No_datasets:',no_train.shape,no_val.shape,no_test.shape)

#组成训练，测试，验证样本
train_files=np.concatenate([have_train,no_train])
validate_files=np.concatenate([have_val,no_val])
test_files=np.concatenate([have_test,no_test])


#在对应的路径下生成文件夹
train_dir='training_data'
val_dir='validation_data'
test_dir='test_data'

os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None



def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            #获得sequence长度
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        #产生0-25次迭代
        progress = IntProgress(min=0, max=size, value=0)
        print(progress,"\n*************************")
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
for fn in log_progress(train_files, name='Training Images'):
    shutil.copy(fn, train_dir)


for fn in log_progress(validate_files, name='Validation Images'):
    shutil.copy(fn, val_dir)

for fn in log_progress(test_files, name='Test Images'):
    shutil.copy(fn, test_dir)


#制作训练样本

IMG_DIM = (224,224)
train_files = glob.glob('training_data/*')
#print(train_files)
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
#print(train_imgs)
train_imgs = np.array(train_imgs)
train_labels = [(fn.split('\\')[1])[0:4] for fn in train_files];
#print(train_labels);
validation_files = glob.glob('validation_data/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [(fn.split('\\')[1])[0:4] for fn in validation_files];
#print(validation_labels);
print('Train dataset shape:', train_imgs.shape,
     '\tValidation dataset shape:', validation_imgs.shape)

print (len(train_labels));

#数据正则化
train_imgs_scaled=train_imgs.astype('float32')
validation_imgs_scaled=validation_imgs.astype('float32')
train_imgs_scaled/=255
validation_imgs_scaled/=255

num_classes = 2
le = LabelEncoder();
le.fit(train_labels)

train_labels_enc = le.transform(train_labels);
validation_labels_enc = le.transform(validation_labels)
#print(train_labels,train_labels_enc,validation_labels,validation_labels_enc);
"""

VGG16分类训练模型


"""

print("***************************************************************************")
input_shape = (224, 224, 3)
vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                  input_shape=input_shape)
output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)
vgg_model.trainable = True
set_trainable = False

for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

print("Trainable layers:", vgg_model.trainable_weights)
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   horizontal_flip=False, vertical_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
input_shape = (224,224,3)
model = Sequential()
model.add(vgg_model)

model.add(Dense(512, input_dim=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])
model.summary()

#set dynamic learning rate and earlystopping
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_acc',
                              patience=10,
                              verbose=1,
                              mode='auto')
#模型训练
history = model.fit_generator(train_generator,steps_per_epoch=50,epochs=50,
                              validation_data=val_generator, validation_steps=20, verbose=1,
                              callbacks=[learning_rate_reduction,early_stopping])

model.save(path+'fine-tune vgg16 with img_aug.h5');
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)




def get_metrics(true_labels, predicted_labels):
    print('Accuracy:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        4))
    print('Precision:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        4))
    print('Recall:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        4))
    print('F1 Score:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        4))


def display_confusion_matrix(true_labels, predicted_labels, classes=[1, 0]):
    total_classes = len(classes)
    level_labels = [total_classes * [0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels,
                                  labels=classes)
    cm_frame = pd.DataFrame(data=cm,
                            columns=pd.MultiIndex(levels=[['Predicted:'], classes],
                                                  labels=level_labels),
                            index=pd.MultiIndex(levels=[['Actual:'], classes],
                                                labels=level_labels))
    print(cm_frame)


def display_classification_report(true_labels, predicted_labels, classes=[1, 0]):
    report = metrics.classification_report(y_true=true_labels,
                                           y_pred=predicted_labels,
                                           labels=classes)
    print(report)


def display_model_performance_metrics(true_labels, predicted_labels, classes=[1, 0]):
    print('Model Performance metrics:')
    print('-' * 30)
    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    print('\nModel Classification report:')
    print('-' * 30)
    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels,
                                  classes=classes)
    print('\nPrediction Confusion Matrix:')
    print('-' * 30)
    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels,
                             classes=classes)


def display_mislabeled_images(dataset, true_labels, predicted_labels, classes=[1, 0]):
    true_labels = true_labels
    a = np.sum([true_labels, predicted_labels.T])
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (224.0, 224.0)
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        plt.subplot(2, num_images, i + 1)
        plt.imshow(dataset[index, :].reshape(224, 224, 3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction label :{0}\n True label :{1}".format(predicted_labels[index],
                                                                   true_labels[index]))


#matplotlib inline


""""
测试
"""

finetune_vgg16_with_img_aug=load_model(path+'fine-tune vgg16 with img_aug.h5')

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
print(time_end-time_start);
predictions = num2class_label_transformer(predictions_enc)

get_metrics(true_labels=test_labels, predicted_labels=predictions)
display_confusion_matrix(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))
display_classification_report(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))
display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))
display_mislabeled_images(dataset=test_imgs_scaled, true_labels=test_labels_enc, predicted_labels=predictions_enc, classes=list(set(test_labels)))