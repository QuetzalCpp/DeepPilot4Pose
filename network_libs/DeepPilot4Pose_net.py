#!/usr/bin/env python3
__author__ = "L. Oyuki Rojas-Perez and Dr. Jose Martinez-Carranza"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "L. Oyuki Rojas-Perez"
__email__ = "carranza@inaoep.mx"

import tensorflow as tf
from keras.layers import Input, Dense, Convolution2D, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import ZeroPadding2D, Dropout, Flatten
from keras.layers import merge, Reshape, Activation, BatchNormalization
from keras.layers import concatenate
from keras import backend as K
from keras.models import Model

import numpy as np
import h5py
import math


def euc_lossx(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.3 * lx)

def euc_lossy(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.3 * lx)

def euc_lossz(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.3 * lx)
    
def euc_lossEMz(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.3 * lx)

def create_DeepPilot4Pose(weights_path=None, tune=False):
    with tf.device('/gpu:0'):
        input = Input(shape=(224, 224, 3))
        
        conv1 = Conv2D(64,(7,7),strides=(2,2), padding='same',activation='relu',name='conv1')(input)
        pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same',name='pool1')(conv1)
        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)
        reduction2 = Conv2D(64,(1,1), padding='same',activation='relu',name='reduction2')(norm1)
        conv2 = Conv2D(192,(3,3),padding='same',activation='relu',name='conv2')(reduction2)
        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)
        pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool2')(norm2)
        icp1_reduction1 = Conv2D(96,(1,1),padding='same',activation='relu',name='icp1_reduction1')(pool2)
        icp1_out1 = Conv2D(128,(3,3),padding='same',activation='relu',name='icp1_out1')(icp1_reduction1)
        icp1_reduction2 = Conv2D(16,(1,1),padding='same',activation='relu',name='icp1_reduction2')(pool2)
        icp1_out2 = Conv2D(32,(5,5),padding='same',activation='relu',name='icp1_out2')(icp1_reduction2)
        icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp1_pool')(pool2)
        icp1_out3 = Conv2D(32,(1,1),padding='same',activation='relu',name='icp1_out3')(icp1_pool)
        icp1_out0 = Conv2D(64,(1,1),padding='same',activation='relu',name='icp1_out0')(pool2)
        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=-1)

        icp2_reduction1 = Conv2D(128,(1,1),padding='same',activation='relu',name='icp2_reduction1')(icp2_in)
        icp2_out1 = Conv2D(192,(3,3),padding='same',activation='relu',name='icp2_out1')(icp2_reduction1)
        icp2_reduction2 = Conv2D(32,(1,1),padding='same',activation='relu',name='icp2_reduction2')(icp2_in)
        icp2_out2 = Conv2D(96,(5,5),padding='same',activation='relu',name='icp2_out2')(icp2_reduction2)
        icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icp2_pool')(icp2_in)
        icp2_out3 = Conv2D(64,(1,1),padding='same',activation='relu',name='icp2_out3')(icp2_pool)
        icp2_out0 = Conv2D(128,(1,1),padding='same',activation='relu',name='icp2_out0')(icp2_in)
        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=-1)
        
        #Branch for X
        icpX_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icpX_in')(icp2_out)
        icpX_reduction1 = Conv2D(96,(1,1),padding='same',activation='relu',name='icpX_reduction1')(icpX_in)
        icpX_out1 = Conv2D(208,(3,3),padding='same',activation='relu',name='icpX_out1')(icpX_reduction1)
        icpX_reduction2 = Conv2D(16,(1,1),padding='same',activation='relu',name='icpX_reduction2')(icpX_in)
        icpX_out2 = Conv2D(48,(5,5),padding='same',activation='relu',name='icpX_out2')(icpX_reduction2)
        icpX_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icpX_pool')(icpX_in)
        icpX_out3 = Conv2D(64,(1,1),padding='same',activation='relu',name='icpX_out3')(icpX_pool)
        icpX_out0 = Conv2D(192,(1,1),padding='same',activation='relu',name='icpX_out0')(icpX_in)
        icpX_out = concatenate([icpX_out0, icpX_out1, icpX_out2, icpX_out3], axis = -1)

        clsX_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='clsx_pool')(icpX_out)
        clsX_reduction_pose = Conv2D(128,(1,1),padding='same',activation='relu',name='clsx_reduction_pose')(clsX_pool)
        clsX_fc1_flat = Flatten()(clsX_reduction_pose)    

        #Branch for Y
        icpY_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icpY_in')(icp2_out)
        icpY_reduction1 = Conv2D(96,(1,1),padding='same',activation='relu',name='icpY_reduction1')(icpY_in)
        icpY_out1 = Conv2D(208,(3,3),padding='same',activation='relu',name='icpY_out1')(icpY_reduction1)
        icpY_reduction2 = Conv2D(16,(1,1),padding='same',activation='relu',name='icpY_reduction2')(icpY_in)
        icpY_out2 = Conv2D(48,(5,5),padding='same',activation='relu',name='icpY_out2')(icpY_reduction2)
        icpY_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icpY_pool')(icpY_in)
        icpY_out3 = Conv2D(64,(1,1),padding='same',activation='relu',name='icpY_out3')(icpY_pool)
        icpY_out0 = Conv2D(192,(1,1),padding='same',activation='relu',name='icpY_out0')(icpY_in)
        icpY_out = concatenate([icpY_out0, icpY_out1, icpY_out2, icpY_out3], axis = -1)

        clsY_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='clsY_pool')(icpY_out)
        clsY_reduction_pose = Conv2D(128,(1,1),padding='same',activation='relu',name='clsY_reduction_pose')(clsY_pool)
        clsY_fc1_flat = Flatten()(clsY_reduction_pose)

        #Branch for Z
        icpZ_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icpZ_in')(icp2_out)
        icpZ_reduction1 = Conv2D(96,(1,1),padding='same',activation='relu',name='icpZ_reduction1')(icpZ_in)
        icpZ_out1 = Conv2D(208,(3,3),padding='same',activation='relu',name='icpZ_out1')(icpZ_reduction1)
        icpZ_reduction2 = Conv2D(16,(1,1),padding='same',activation='relu',name='icpZ_reduction2')(icpZ_in)
        icpZ_out2 = Conv2D(48,(5,5),padding='same',activation='relu',name='icpZ_out2')(icpZ_reduction2)
        icpZ_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icpZ_pool')(icpZ_in)
        icpZ_out3 = Conv2D(64,(1,1),padding='same',activation='relu',name='icpZ_out3')(icpZ_pool)
        icpZ_out0 = Conv2D(192,(1,1),padding='same',activation='relu',name='icpZ_out0')(icpZ_in)
        icpZ_out = concatenate([icpZ_out0, icpZ_out1, icpZ_out2, icpZ_out3], axis = -1)

        clsZ_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='clsZ_pool')(icpZ_out)
        clsZ_reduction_pose = Conv2D(128,(1,1),padding='same',activation='relu',name='clsZ_reduction_pose')(clsZ_pool)
        clsZ_fc1_flat = Flatten()(clsZ_reduction_pose)
        
        #Branch for exponential Map
        icpO_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='icpO_in')(icp2_out)
        icpO_reduction1 = Conv2D(96,(1,1),padding='same',activation='relu',name='icpO_reduction1')(icpO_in)
        icpO_out1 = Conv2D(208,(3,3),padding='same',activation='relu',name='icpO_out1')(icpO_reduction1)
        icpO_reduction2 = Conv2D(16,(1,1),padding='same',activation='relu',name='icpO_reduction2')(icpO_in)
        icpO_out2 = Conv2D(48,(5,5),padding='same',activation='relu',name='icpO_out2')(icpO_reduction2)
        icpO_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='icpO_pool')(icpO_in)
        icpO_out3 = Conv2D(64,(1,1),padding='same',activation='relu',name='icpO_out3')(icpO_pool)
        icpO_out0 = Conv2D(192,(1,1),padding='same',activation='relu',name='icpO_out0')(icpO_in)
        icpO_out = concatenate([icpO_out0, icpO_out1, icpO_out2, icpO_out3], axis = -1)

        clsO_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='clsO_pool')(icpO_out)
        clsO_reduction_pose = Conv2D(128,(1,1),padding='same',activation='relu',name='clsO_reduction_pose')(clsO_pool)
        clsO_fc1_flat = Flatten()(clsO_reduction_pose)

        cls1_branchX_pose = Dense(1024,activation='relu',name='cls1_branchX_pose')(clsX_fc1_flat)
        cls1_branchY_pose = Dense(1024,activation='relu',name='cls1_branchY_pose')(clsY_fc1_flat)
        cls1_branchZ_pose = Dense(1024,activation='relu',name='cls1_branchZ_pose')(clsZ_fc1_flat)
        cls1_branchO_pose = Dense(1024,activation='relu',name='cls1_branchO_pose')(clsO_fc1_flat)

        #----------------------------------------

        pose_x = Dense(1,name='pose_x')(cls1_branchX_pose)
        pose_y = Dense(1,name='pose_y')(cls1_branchY_pose)
        pose_z = Dense(1,name='pose_z')(cls1_branchZ_pose)
        EMz = Dense(1,name='EMz')(cls1_branchO_pose)
       
        net = Model(inputs=input, outputs=[pose_x, pose_y, pose_z, EMz])

    if tune:
        if weights_path:
            weights_data = np.load(weights_path, encoding = 'latin1', allow_pickle = True).item()
            for layer in net.layers:
                if layer.name in weights_data.keys():
                    layer_weights = weights_data[layer.name]
                    layer.set_weights((layer_weights['weights'], layer_weights['biases']))
            print("FINISHED SETTING THE WEIGHTS!")
    return net
