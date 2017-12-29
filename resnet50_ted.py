import os, json
from glob import glob

from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras.models import Sequential, Model
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Input, Activation, Add
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing import image
from keras.utils import get_file, layer_utils #???
from keras.applications.imagenet_utils import preprocess_input

from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.applications.resnet50 import identity_block, conv_block

from keras import backend as K
# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)
K.set_image_data_format('channels_first') # Ex: (3, 224, 224)



class Resnet50():
    """The Resnet 50 Imagenet model"""

    
    def __init__(self, size=(224,224), include_top=True):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.VGG_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))  # vgg_mean (in BGR)
        self.VGG_DROPOUT = 0.5
        self.dropout = self.VGG_DROPOUT
        self._create(size, include_top)
        self._get_classes()
        
        
    def _vgg_preprocess(self, x):
        """
            Subtracts the mean RGB value, and transposes RGB to BGR.
            The mean RGB was computed on the image set used to train the VGG model
            (VGG-16 and VGG-19 were trained using Caffe, and Caffe uses OpenCV to load images which uses BGR by default, so both VGG models             are expecting BGR images.)
        
            Args:
                x: Image array (height x width x channels)
            Returns:
                Image array (height x width x transposed_channels)
        """
        x = x - self.VGG_MEAN
        return x[:,::-1] # reverse axis RGB into BGR
        
        
    def _get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it's not already in the cache.
        """
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        
    
    def predict(self, imgs, details=False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = np.array([all_preds[i][idxs[i]] for i in range(len(idxs))])
        classes = [self.classes[idx] for idx in idxs]
        return preds, idxs, classes
    
    
    def _create(self, size, include_top):
        input_shape = (3,)+size
        img_input = Input(shape=input_shape)
        bn_axis = 1
        
        x = Lambda(self._vgg_preprocess)(img_input)
        x = ZeroPadding2D((3,3))(x)
        
        x = Conv2D(64, (7,7), strides=(2,2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3,3), strides=(2,2))(x)
        
        x = conv_block(x, 3, filters=[64,64,256], stage=2, block='a', strides=(1,1))
        x = identity_block(x, 3, filters=[64,64,256], stage=2, block='b')
        x = identity_block(x, 3, filters=[64,64,256], stage=2, block='c')
        
        x = conv_block(x, 3, filters=[128,128,512], stage=3, block='a', strides=(2,2))
        for b in ['b', 'c', 'd']: x = identity_block(x, 3, filters=[128,128,512], stage=3, block=b)
        
        x = conv_block(x, 3, filters=[256,256,1024], stage=4, block='a', strides=(2,2))
        for b in ['b', 'c', 'd', 'e', 'f']: x = identity_block(x, 3, filters=[256,256,1024], stage=4, block=b)
        
        x = conv_block(x, 3, filters=[512,512,2048], stage=2, block='a', strides=(2,2))
        x = identity_block(x, 3, filters=[512,512,2048], stage=5, block='b')
        x = identity_block(x, 3, filters=[512,512,2048], stage=5, block='c')
        
        if include_top:
            x = AveragePooling2D(pool_size=(7,7), name='avg_pool', padding='valid')(x)
            x = Flatten()(x)
            x = Dense(1000, activation='softmax', name='fc1000')(x)
            fname = 'resnet50.h5'
        else:
            fname = 'resnet_nt.h5'
        
        self.model = Model(inputs=img_input, outputs=x, name='ResNet50')
        #convert_all_kernels_in_model(self.model)
        self.model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
        
        
    def get_batches(self, path, gen=image.ImageDataGenerator(), class_mode='categorical', shuffle=True, 
                    batch_size=8, target_size=(224,224)):
        return gen.flow_from_directory(path, target_size=target_size, class_mode=class_mode, 
                                       shuffle=shuffle, batch_size=batch_size)

    
    def finetune(self, batches):
        model = self.model
        model.layers.pop() #Sequential: model.pop()
        for layer in model.layers: layer.trainable = False
        m = Dense(batches.num_classes, activation='softmax')(model.layers[-1].output)
        self.model = Model(model.input, m)
        self.model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    def fit(self, batches, val_batches, epochs=1, verbose=2):
        self.model.fit_generator(batches, epochs=epochs, verbose=verbose,
                                 steps_per_epoch=ceil(batches.n/batches.batch_size),
                                 validation_data=val_batches,
                                 validation_steps=ceil(val_batches.n/val_batches.batch_size))
        
    
    def test(self, path, batch_size=8):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch
            
            See Keras documentation: https://keras.io/models/model/#predict_generator
            
            Args:
                path (string) :  Path to the target directory. It should contain 
                                one subdirectory per class.
                batch_size (int) : The number of images to be considered in each batch.
            
            Returns:
                test_batches 
                numpy array(s) of predictions for the test batches.
        """
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, steps=ceil(test_batches.n/test_batches.batch_size))
    
    
    
    
    
   