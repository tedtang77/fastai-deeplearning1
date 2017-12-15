import math, os, sys, json, re

import pickle

from glob import glob
import numpy as np
from numpy.random import normal, permutation, uniform, choice
from matplotlib import pyplot as plt

from operator import itemgetter, attrgetter, methodcaller

from collections import OrderedDict
import itertools
from itertools import chain
import bcolz
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from keras.utils import get_file, to_categorical
from keras.preprocessing import image, sequence
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, SimpleRNN
from keras.layers import Input, Embedding, Dot, dot, Add, add, Concatenate, SpatialDropout1D
from keras.layers import merge # [Deprecared] merge 
from keras.layers import TimeDistributed, Activation
from keras.layers.core import Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
from keras.metrics import categorical_accuracy as accuracy
from keras.metrics import categorical_crossentropy as crossentropy

from keras import backend as K
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)
K.set_image_data_format('channels_first')

from keras.metrics import categorical_accuracy as accuracy
from keras.metrics import categorical_crossentropy as crossentropy

#from vgg16 import *
#from vgg16bn import *

def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, 
                target_size=(224, 224), class_mode='categorical'):
    return gen.flow_from_directory(path, target_size=target_size, 
                                   class_mode=class_mode, batch_size=batch_size, shuffle=shuffle)


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1) 
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1) 
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1) 
    return (batches.classes, val_batches.classes, 
            onehot(batches.classes), onehot(val_batches.classes),
            batches.filenames, val_batches.filenames, test_batches.filenames)


def get_data(path, target_size=(224, 224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])


def split_at(model, layer_type):
    """
    Split model into two layer groups in different layer types
    
    Arguments:
        model: type of keras.models.Model (ex: from VGG16 or Vgg16BN)
        layer_type: layer type of keras.layers (ex: Conv2D)
    
    Returns:
        first_group_layers: all layers in model before last layer_type
        second_group_layers: all layers in model after last layer_type
    """
    
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]


def get_split_models(model):
    """
        Gets the two models spliting convolution model and dense model at Flatten layer
            
        Returns:
        conv_model: the model constructing by Vgg convolution layers ending at the last MaxPooling2D layer 
        fc_model: the model constructing by Vgg dense layers starting at Flatten layer
            
    """
    flatten_idx = [idx for idx, layer in enumerate(model.layers) if type(layer)==Flatten][0]
        
    conv_model = Sequential(model.layers[:flatten_idx])
    for layer in conv_model.layers: layer.trainable = False
    conv_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    fc_model = Sequential([ 
            Flatten(input_shape=conv_model.layers[-1].output_shape[1:]) 
        ])
    for layer in model.layers[flatten_idx+1:]: 
        fc_model.add(layer)
    for layer in fc_model.layers: layer.trainable = True
    fc_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return conv_model, fc_model


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting 'normalize=True'
        (This function is copied from scikit docs: https://github.com/scikit-learn/scikit-learn/blob/master/examples/model_selection/plot_confusion_matrix.py)
        
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def onehot(x):
    return to_categorical(x)
    #return np.array(OneHotEncoder().fit_transform(np.expand_dims(x, axis=1)).todense())

    
def vgg_ft(out_dim):
    vgg = Vgg16()
    vgg.ft(out_dim)
    return vgg.model
    
    
def vggbn_ft(out_dim):
    vgg = Vgg16BN()
    vgg.ft(out_dim)
    return vgg.model


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    
    
def load_array(fname):
    return bcolz.open(fname)[:]


def eval_accuracy(labels, preds):
    """
        https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
    """
   
    acc_value = accuracy(labels, preds)
    with sess.as_default():
        eval_result = acc_value.eval()
    return eval_result.mean()


def eval_crossentropy(labels, preds):
    """
        
        Ref: https://stackoverflow.com/questions/46687064/categorical-crossentropy-loss-no-attribute-get-shape
    """
   
    entropy_value = crossentropy(K.constant(labels.astype('float32')), K.constant(preds.astype('float32')))
    with sess.as_default():
        eval_result = entropy_value.eval()
    return eval_result.mean()


def ceil(x):
    return int(math.ceil(x))


def floor(x):
    return int(math.floor(x))


class MixIterator(object):
    
    def __init__(self, iters):
        self.iters = iters
        self.n = sum([itr.n for itr in self.iters])
        self.batch_size = sum([itr.batch_size for itr in self.iters])
    
    def reset(self):
        for itr in self.iters: itr.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self, *args, **kwargs):
        nexts = [next(itr) for itr in self.iters]
        n0 = np.concatenate([n[0] for n in nexts])
        n1 = np.concatenate([n[1] for n in nexts])
        return (n0, n1)





