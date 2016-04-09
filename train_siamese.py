from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy
import os
import VIPerDS
import siamese
import random
#import keras.backend as T
import theano
import theano.tensor as T
from siamese_model import build_model


# loss basata su distanza
# DOPO  per classificare o nearest neighbour o K-means con K=2
def my_siamese_loss(y_true, y_pred):

    v_pari= y_pred[0::2]
    v_dispari= y_pred[1::2]
    y_pari= y_true[0::2]
    y_dispari= y_true[1::2]
    d=T.square(v_pari-v_dispari)
    l=T.sum(d,axis=1)
    loss=T.mean(T.transpose(y_pari) * l + T.transpose(1-y_pari)*T.maximum(margin-l,0))

    return loss


def coupling_dataset(ds_set, label, num_couple):
    # 0 = diverso
    # 1 = uguale

    male = ds_set[label==1]
    female = ds_set[label==-1]

    shape = list(ds_set.shape)
    shape[0] = num_couple
    new_set = numpy.empty(shape=shape)
    shape = list(label.shape)
    shape[0] = num_couple
    new_lab = numpy.empty(shape=shape)

    for i in range(0,num_couple,2):
        i1 = random.randint(0, len(label)-1)
        i2 = random.randint(0, len(label)-1)

        # new_set = numpy.vstack((new_set, ds_set[i1,:,:]))
        # new_set = numpy.vstack((new_set, ds_set[i2,:,:]))
        new_set[i,:,:,:] = ds_set[i1,:,:,:]
        new_set[i+1,:,:,:] = ds_set[i2,:,:,:]

        lab = int(label[i1] == label[i2])
        new_lab[i] = lab
        new_lab[i+1] = lab

    return new_set, new_lab




video_directory = "/home/cla/Downloads/CMD/CMD/CMD"

directory_list = []
for direct in os.listdir(video_directory):
    if not os.path.isdir(video_directory+'/'+direct):
        continue
    else:
        directory_list.append(video_directory+'/'+direct)


def load_pkl_video(dir):
    import cPickle as pickle
    import sys
    sys.setrecursionlimit(10000)

    file_load = open(dir + "/prova.pkl", 'rb')
    m = pickle.load(file_load)
    return m


def load_train_samples(x, y, dirs, maxsize=None):
    for d in random.sample(dirs,len(dirs)):
            (new_img, new_labels) = load_pkl_video(d)
            # corrected_labels = map(lambda x: 0 if x < 0 else x, new_labels)
            # labels = np_utils.to_categorical(corrected_labels, nb_classes)
            # y = numpy.vstack([y, labels])
            y = numpy.append(y, new_labels)
            x = numpy.vstack([x, new_img])
            if maxsize and y.shape[0] > maxsize:
                break
    return (x,y)

random.seed(2134)
random.shuffle(directory_list)

val_list= directory_list[16:]
train_list = directory_list[:16]
# train_list = directory_list

margin = 1

batch_size = 32
# nb_classes = 1
nb_classes = 2
nb_epoch = 1

data_augmentation = False

# input image dimensions
img_rows, img_cols = 128, 48
# the VIPerDS images are RGB
img_channels = 3

# (X_train, y_train), (X_test, y_test) = VIPerDS.load_data()
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
X_test = numpy.empty([0, 3, 128, 48], dtype='float32')
Y_test = numpy.empty([0, 1])
X_train = numpy.empty([0, 3, 128, 48], dtype='float32')
Y_train = numpy.empty([0, 1])
(X_test, Y_test) = load_train_samples(X_test,Y_test, val_list)
X_test, Y_test = coupling_dataset(X_test, Y_test, 25000)


# model_name = 'my_model_architecture.json'
# model_weights = 'my_model_weights.h5'
#
# m = model_from_json(open(model_name).read())
# if model_weights:
#     m.load_weights(model_weights)


# m = Sequential()

# m.add(Convolution2D(32, 5, 5, border_mode='valid',
#                         input_shape=(img_channels, img_rows, img_cols)))
# m.add(Activation('relu'))
# m.add(Convolution2D(32, 5, 5))
# m.add(Activation('relu'))
# m.add(MaxPooling2D(pool_size=(2, 2)))
# # m.add(ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
# m.add(Dropout(0.25))

# m.add(Convolution2D(64, 3, 3, border_mode='same'))
# m.add(Activation('relu'))
# m.add(Convolution2D(64, 3, 3))
# m.add(Activation('relu'))
# m.add(MaxPooling2D(pool_size=(2, 2)))
# m.add(Dropout(0.25))

# m.add(Convolution2D(128, 3, 3, border_mode='same'))
# m.add(Activation('relu'))
# m.add(Convolution2D(128, 3, 3))
# m.add(Activation('relu'))
# m.add(MaxPooling2D(pool_size=(2, 2)))
# m.add(Dropout(0.25))

# m.add(Flatten())
# m.add(Dense(64))
# m.add(Activation('relu'))
# #m.add(Dropout(0.5))

# last_layer = 32
# m.add(Dense(last_layer))
# m.add(Activation('relu'))
# # #m.add(Dropout(0.5))

# m.load_weights("my_model_weights.h5")

# earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min')

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# m.compile(loss=my_siamese_loss, optimizer=sgd)

m = build_model((img_channels, img_rows, img_cols))

if not data_augmentation:
    print('Not using data augmentation.')


#cambio il seed per caricare gli esempi in modi diversi
random.seed(45907)

trainsize = 15000
discardfrom = 12000
couples = 15000

# for i in range(1,10):
#     if Y_train.shape[0] < trainsize:
#         (X_train, Y_train) = load_train_samples(X_train, Y_train, train_list, trainsize)
#     else:
#         # X_train = X_train[discardfrom:, :]
#         # Y_train = Y_train[discardfrom:, :]
#         X_train = X_train[discardfrom:, :]
#         Y_train = Y_train[discardfrom:]
#         (X_train, Y_train) = load_train_samples(X_train, Y_train, train_list, trainsize)
#
#     X_train, Y_train = coupling_dataset(X_train, Y_train, couples)
#
#     m.fit(X_train, Y_train, batch_size=batch_size,
#           nb_epoch=nb_epoch, show_accuracy=True,
#           validation_data=(X_test, Y_test), shuffle=False
#           # ,callbacks=[earlyStopping]
#           )

(X_train, Y_train) = load_train_samples(X_train, Y_train, train_list, trainsize)
X_train, Y_train = coupling_dataset(X_train, Y_train, couples)

m.fit(X_train, Y_train, batch_size=batch_size,
      nb_epoch=nb_epoch, show_accuracy=True,
      validation_data=(X_test, Y_test), shuffle=False
      # ,callbacks=[earlyStopping]
      )

json_string = m.to_json()
open('my_model_architecture.json', 'w').write(json_string)
m.save_weights('my_model_weights.h5'
                   , overwrite=True
                   )
