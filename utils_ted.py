import math, os, sys, json, re

import pickle

from glob import glob
import numpy as np
from numpy.random import normal, permutation, uniform
from matplotlib import pyplot as plt

from operator import itemgetter, attrgetter, methodcaller

from collections import OrderedDict
import itertools
from itertools import chain
import bcolz
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from keras.utils import get_file
from keras.preprocessing import image, sequence
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM
from keras.layers import Input, Embedding, Dot, dot, add, Concatenate, SpatialDropout1D
from keras.layers import merge # [Deprecared] merge 
from keras.layers.core import Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)
K.set_image_data_format('channels_first')

from keras.metrics import categorical_accuracy as accuracy
from keras.metrics import categorical_crossentropy as crossentropy

from vgg16 import *
from vgg16bn import *

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









