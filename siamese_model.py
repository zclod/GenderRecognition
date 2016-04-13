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

def my_siamese_loss(y_true, y_pred):

    v_pari= y_pred[0::2]
    v_dispari= y_pred[1::2]
    y_pari= y_true[0::2]
    y_dispari= y_true[1::2]
    d=T.square(v_pari-v_dispari)
    l=T.sum(d,axis=1)
    loss=T.mean(T.transpose(y_pari) * l + T.transpose(1-y_pari)*T.maximum(margin-l,0))

    return loss

margin = 1
# weights_path = "my_model_weights.h5"

def build_model(input_shape, weights=None):
    m = Sequential()

    m.add(Convolution2D(32, 5, 5, border_mode='valid',
                            input_shape=input_shape))
    # m.add(Activation('relu'))
    m.add(Convolution2D(32, 5, 5))
    # m.add(Activation('relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    # m.add(ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    m.add(Dropout(0.25))

    m.add(Convolution2D(64, 3, 3, border_mode='same'))
    # m.add(Activation('relu'))
    m.add(Convolution2D(64, 3, 3))
    # m.add(Activation('relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.25))

    m.add(Convolution2D(128, 3, 3, border_mode='same'))
    # m.add(Activation('relu'))
    m.add(Convolution2D(128, 3, 3))
    m.add(Activation('sigmoid'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.25))

    m.add(Flatten())
    m.add(Dense(64))
    # m.add(Activation('relu'))
    #m.add(Dropout(0.5))

    last_layer = 32
    m.add(Dense(last_layer))
    # m.add(Activation('relu'))
    # #m.add(Dropout(0.5))

    if weights:
        m.load_weights(weights)

    # earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min')

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    m.compile(loss=my_siamese_loss, optimizer=sgd)

    return m
