""" 
 @author   Maksim Penkin
https://github.com/MaksimPenkin

"""

import os
import sys 

import glob
import time
import random
import datetime
from datetime import datetime
import numpy as np
import pandas as pd

import skimage
from skimage.io import imread, imsave
import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops

import progressbar as pb
import math
import shutil

from util.util import WriteToCSVFile, WriteToTXTFile, Meter, data_normaliser

def rnn(step_function, inputs, initial_states):
    ndim = len(inputs.get_shape())
    if ndim < 3:
        print('Runtime error: def rnn(): Input should be at least 3D.')
        exit(1)

    # Transpose to time-major, i.e.
    # from (batch, time, ...) to (time, batch, ...)
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))
    states = tuple(initial_states)

    time_steps = tf.shape(inputs)[0]
    outputs, _ = step_function(inputs[0], initial_states)
    output_ta = tensor_array_ops.TensorArray(
        dtype=outputs.dtype,
        size=time_steps,
        tensor_array_name='output_ta')
    input_ta = tensor_array_ops.TensorArray(
        dtype=inputs.dtype,
        size=time_steps,
        tensor_array_name='input_ta')
    input_ta = input_ta.unstack(inputs)
    time = tf.constant(0, dtype='int32', name='time')

    def step(time, output_ta_t, *states):
        current_input = input_ta.read(time)
        output, new_states = step_function(current_input,
                                            tuple(states))
        for state, new_state in zip(states, new_states):
            new_state.set_shape(state.get_shape())
        output_ta_t = output_ta_t.write(time, output)
        return (time + 1, output_ta_t) + tuple(new_states)

    final_outputs = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=step,
        loop_vars=(time, output_ta) + states,
        parallel_iterations=32,
        swap_memory=True)
    last_time = final_outputs[0]
    output_ta = final_outputs[1]
    new_states = final_outputs[2:]

    outputs = output_ta.stack()
    last_output = output_ta.read(last_time - 1)

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)

    return last_output, outputs, new_states

def lrnn(X, W, horizontal=True, reverse=False):
    def reorder_input(X):
        # X.shape = (batch_size, row, column, channel)
        if horizontal:
            X = tf.transpose(X, [0, 2, 1, 3])
        if reverse:
            X = tf.reverse(X, [1])
        return X

    def reorder_output(X):
        if reverse:
            X = tf.reverse(X, [1])
        if horizontal:
            X = tf.transpose(X, [0, 2, 1, 3])
        return X
    
    X = reorder_input(X)
    W = reorder_input(W)
    def compute(x, a):
        Yt = a[0]
        Xt = x[:, :, :tf.shape(x)[-1] // 2 ]
        Wt = x[:, :,  tf.shape(x)[-1] // 2:]
        Yt = Wt * (Yt - Xt) + Xt # y*w + (1-w)*x
        return Yt, [Yt]
    initializer = tf.zeros_like(X[:, 0])
    S = rnn(compute, tf.concat([X, W], axis=-1), [initializer])[1]
    Y = reorder_output(S)
    return Y

def attention_module(inp1, inp2, inp3, reuse=None, trainable=True, is_training=True, scope=None):
    # inp1, inp2, inp3 are tensors of shape (B,H,W,C)
    # Reshape inp1 to (B, C, H*W)
    # Reshape inp2 to (B, C, H*W) and then transpose to get (B, H*W, C)
    # Multiply inp1 and inp2 to get correlation feature matrix: (B, C, C)
    # Reshape inp3 to (B, C, H*W) and implement attention to channels based on correlation between features

    with tf.compat.v1.variable_scope(scope):
        inp1 = tf.compat.v1.layers.batch_normalization(inp1, training=is_training, trainable=trainable, reuse=reuse)
        inp2 = tf.compat.v1.layers.batch_normalization(inp2, training=is_training, trainable=trainable, reuse=reuse)

        inp1_reshaped = tf.reshape(inp1,[-1, tf.shape(inp1)[-1], tf.shape(inp1)[1]*tf.shape(inp1)[2]])
        inp2_reshaped = tf.transpose(tf.reshape(inp2,[-1, tf.shape(inp2)[-1], tf.shape(inp2)[1]*tf.shape(inp2)[2]]), perm=[0,2,1])
        inp3_reshaped = tf.reshape(inp3,[-1, tf.shape(inp3)[-1], tf.shape(inp3)[1]*tf.shape(inp3)[2]])

        cov_tensor = tf.nn.softmax(tf.matmul(inp1_reshaped,inp2_reshaped), axis=-1, name='feature_covariance')

        attention_tensor = tf.matmul(cov_tensor,inp3_reshaped, name='attention_map')
        attention_out = tf.reshape(attention_tensor, [-1, tf.shape(inp3)[1],tf.shape(inp3)[2],tf.shape(inp3)[3]], name='attention_filtered')

    return attention_out

def conv2d(inputs, dim, ksize=3, stride=1, activation=tf.nn.relu, normalizer_fn=None, normalizer_params=None, regularizator=None, reuse=None, trainable=True, scope=None):
    # slim.l2_regularizer(scale=0.01)
    return slim.conv2d(
        inputs,
        num_outputs=dim,
        kernel_size=ksize,
        stride=stride,
        padding='SAME',
        data_format='NHWC',
        rate=1,
        activation_fn=activation,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params,
        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
        weights_regularizer=regularizator,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=reuse,
        trainable=trainable,
        scope=scope
    )

def adaptive_global_average_pool_2d(x):
    c = x.get_shape()[-1]
    return tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))

def channel_attention(inputs, dim, reduction, activation=tf.nn.relu, regularizator=None, reuse=None, trainable=True, scope=None, is_training=True):
    with tf.compat.v1.variable_scope(scope):
        skip_conn = tf.identity(inputs, name='identity')

        x = tf.compat.v1.layers.batch_normalization(inputs, training=is_training, trainable=trainable, reuse=reuse)

        x = adaptive_global_average_pool_2d(x)

        x = conv2d(x, dim//reduction, ksize=1, regularizator=regularizator, reuse=reuse, trainable=trainable)
        x = conv2d(x, dim, ksize=1, activation=None, regularizator=regularizator, reuse=reuse, trainable=trainable)
        x = tf.nn.sigmoid(x)

        return tf.multiply(skip_conn, x)

def conv2d_bn(inputs, dim, ksize=3, stride=1, activation=tf.nn.relu, regularizator=None, reuse=None, trainable=True, scope=None, is_training=True):
    with tf.compat.v1.variable_scope(scope):
        x = conv2d(inputs, dim, ksize=ksize, stride=stride, activation=None, regularizator=regularizator, reuse=reuse, trainable=trainable, scope='conv')
        x = tf.compat.v1.layers.batch_normalization(x, training=is_training, trainable=trainable, reuse=reuse)
        x = tf.nn.relu(x)
        return x

def depthwise_conv2d(inputs, ksize=3, stride=1, activation=tf.nn.relu, normalizer_fn=None, normalizer_params=None, regularizator=None, reuse=None, trainable=True, scope=None):
    return slim.separable_conv2d(
        inputs,
        num_outputs=None,
        kernel_size=ksize,
        depth_multiplier=1,
        stride=stride,
        padding='SAME',
        data_format='NHWC',
        rate=1,
        activation_fn=activation,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params,
        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
        pointwise_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
        weights_regularizer=regularizator,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=reuse,
        trainable=trainable,
        scope=scope
    )

def separable_conv2d(inputs, dim, ksize=3, stride=1, activation=tf.nn.relu, normalizer_fn=None, normalizer_params=None, regularizator=None, reuse=None, trainable=True, scope=None):
    return slim.separable_conv2d(
        inputs,
        num_outputs=dim,
        kernel_size=ksize,
        depth_multiplier=1,
        stride=stride,
        padding='SAME',
        data_format='NHWC',
        rate=1,
        activation_fn=activation,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params,
        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
        pointwise_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
        weights_regularizer=regularizator,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=reuse,
        trainable=trainable,
        scope=scope
    )

def ResBlock(inputs, dim, ksize=3, stride=1, activation=tf.nn.relu, regularizator=None, reuse=None, trainable=True, scope=None):
    with tf.compat.v1.variable_scope(scope):
        net = conv2d(inputs, dim, ksize=ksize, stride=stride, activation=activation, regularizator=regularizator, reuse=reuse, trainable=trainable, scope='conv1')
        net = conv2d(net, dim, ksize=ksize, stride=stride, activation=None, regularizator=regularizator, reuse=reuse, trainable=trainable, scope='conv2')
    return net + inputs

def MobileNetv2ResBlock(inputs, dim, factor, ksize=3, stride=1, activation=tf.nn.relu, regularizator=None, reuse=None, trainable=True, scope=None, is_training=True):
    with tf.compat.v1.variable_scope(scope):
        net = conv2d(inputs, dim*factor, ksize=1, stride=stride, activation=activation,  regularizator=regularizator, reuse=reuse, trainable=trainable, scope='pw1')
        net = depthwise_conv2d(net, ksize=ksize, stride=stride, activation=None,  regularizator=regularizator, reuse=reuse, trainable=trainable, scope='dw')
        net = conv2d(net, dim, ksize=1, stride=stride, activation=activation,  regularizator=regularizator, reuse=reuse, trainable=trainable, scope='pw2')
    return net + inputs

def MobileNetResBlock(inputs, dim, ksize=3, stride=1, activation=tf.nn.relu, regularizator=None, reuse=None, trainable=True, scope=None, is_training=True):
    with tf.compat.v1.variable_scope(scope):
        net = separable_conv2d(inputs, dim, ksize=1, stride=stride, activation=activation,  regularizator=regularizator, reuse=reuse, trainable=trainable, scope='sepconv')
    return net + inputs

def residual_channel_attention_block(inputs, dim, ksize=3, stride=1, activation=tf.nn.relu, regularizator=None, reuse=None, trainable=True, scope=None, is_training=True):
    with tf.compat.v1.variable_scope(scope):
        skip_conn = tf.identity(inputs, name='identity')

        x = conv2d(inputs, dim, ksize=ksize, stride=stride, activation=activation, regularizator=regularizator, reuse=reuse, trainable=trainable, scope='conv1')
        x = conv2d(x, dim, ksize=ksize, stride=stride, activation=None, regularizator=regularizator, reuse=reuse, trainable=trainable, scope='conv2')
        
        x = channel_attention(x, dim, reduction=2, activation=activation, regularizator=regularizator, reuse=reuse, trainable=trainable, scope="CA", is_training=is_training)
        return x + skip_conn 

def ScaleRotate(img, angle, scale):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), angle, scale)
    return cv2.warpAffine(img, M, (cols, rows), flags = cv2.INTER_NEAREST)[..., np.newaxis]

class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.max_value = self.args.max_value
        self.normalizer_data = self.args.normalizer_data
        if self.normalizer_data not in ['maxmin', 'standart_scaler', 'byvalue']:
            print('Runtime error: self.args.normalizer_data can be only [maxmin, standart_scaler, byvalue]')
            exit(1)
        self.normalizer_data_params = {'sub':0., 'scale':1.}
        if self.normalizer_data == 'byvalue':
            self.normalizer_data_params = {'sub':self.args.normalizer_data_sub, 'scale':self.args.normalizer_data_scale}
            
        self.chns = 1
        self.cnn_size = self.args.cnn_size
        self.feature_size = self.args.feature_size
        self.conv_blocks_num = self.args.conv_blocks_num
        if self.conv_blocks_num % 2 != 0:
            print('Runtime error: conv_blocks_num should be even!')
            exit(1)
        self.H_conv = (self.conv_blocks_num//2 + 2) * [None]
        
        self.train_dir = args.checkpoints
        self.model_name = 'degibbs.model'
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.reg_scale = args.reg_scale

        if args.phase == 'train':
            self.crop_size = 48
            self.num_random_crops = self.args.num_random_crops
            self.buffer_size = self.num_random_crops * 400
            self.epoch = args.epoch
            self.learning_rate = args.learning_rate

            self.data_root = os.path.dirname(args.datalist)
            self.val_data_root = os.path.dirname(args.val_datalist)

            self.data_list = open(args.datalist, 'rt').read().splitlines()
            self.data_list = list(map(lambda x: x.split(' '), self.data_list))

            self.val_data_list = open(args.val_datalist, 'rt').read().splitlines()
            self.val_data_list = np.array(list(map(lambda x: x.split(' '), self.val_data_list)))

            random.shuffle(self.data_list)
            self.data_size = (len(self.data_list) * self.num_random_crops) // self.batch_size
            self.max_steps = int(self.epoch * self.data_size)

            if (self.args.restore_ckpt == 0):
                WriteToCSVFile(os.path.join(self.train_dir, 'train.csv'), [['Step', 'Loss', 'RMSE', 'MAE', 'PSNR']], mode='a')
                WriteToCSVFile(os.path.join(self.train_dir, 'val.csv'), [['Step', 'Loss', 'RMSE', 'MAE', 'PSNR']], mode='a')
        elif args.phase == 'test':
            self.test_data_root = os.path.dirname(args.test_datalist)

            self.test_data_list = open(args.test_datalist, 'rt').read().splitlines()
            self.test_data_list = np.array(list(map(lambda x: x.split(' '), self.test_data_list)))

            if self.args.output_path != "":
                if not os.path.exists(self.args.output_path):
                    os.makedirs(self.args.output_path)
                self.val_path_gt_save = os.path.join(self.args.output_path, 'gt')
                self.val_path_gibbs_save = os.path.join(self.args.output_path, 'gibbs')
                self.val_path_degibbs_save = os.path.join(self.args.output_path, 'degibbs')
                self.val_path_error_save = os.path.join(self.args.output_path, 'error')
                if not os.path.exists(self.val_path_gt_save):
                    os.makedirs(self.val_path_gt_save)
                if not os.path.exists(self.val_path_gibbs_save):
                    os.makedirs(self.val_path_gibbs_save)
                if not os.path.exists(self.val_path_degibbs_save):
                    os.makedirs(self.val_path_degibbs_save)
                if not os.path.exists(self.val_path_error_save):
                    os.makedirs(self.val_path_error_save)
            WriteToCSVFile(os.path.join(self.train_dir, 'test.csv'), [['Image', 'Loss', 'RMSE', 'MAE', 'PSNR']], mode='a')

        else:
            print('Runtime error: args.phase can be only [train, test]')
            exit(1)
    
    def generator(self, inputs, mode='train', reuse=False, scope='g_net'):
        var_reuse = reuse
        var_trainable=False
        if (mode=='train'):
            var_trainable=True
        with tf.compat.v1.variable_scope(scope):
            conv_pre = conv2d(inputs, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)

            conv1 = residual_channel_attention_block(conv_pre, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN1')
            conv2 = residual_channel_attention_block(conv1, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN2')
            conv3 = residual_channel_attention_block(conv2, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN3')
            conv4 = residual_channel_attention_block(conv3, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN4')
            conv5 = residual_channel_attention_block(conv4, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN5')
            conv6 = residual_channel_attention_block(conv5, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN6')
            conv7 = residual_channel_attention_block(conv6, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN7')
            
            conv8 = residual_channel_attention_block(conv7, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN8') + conv6
            conv9 = residual_channel_attention_block(conv8, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN9') + conv5
            conv10 = residual_channel_attention_block(conv9, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN10') + conv4
            conv11 = residual_channel_attention_block(conv10, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN11') + conv3
            conv12 = residual_channel_attention_block(conv11, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN12') + conv2
            conv13 = residual_channel_attention_block(conv12, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN13') + conv1
            conv14 = residual_channel_attention_block(conv13, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN14') + conv_pre

            weights_gen = conv2d(inputs, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, activation=tf.nn.relu, trainable=var_trainable)
            weights_gen = conv2d(weights_gen, 128, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), activation=tf.nn.relu, reuse=var_reuse, trainable=var_trainable)
            weights = conv2d(weights_gen, 256, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), activation=tf.nn.leaky_relu, reuse=var_reuse, trainable=var_trainable)

            features = conv2d(conv14, 32, 1, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)

            g_RNN1_x1 = weights[:, :, :,   0: 32]
            g_RNN1_y1 = weights[:, :, :,  32: 64]
            g_RNN1_x2 = weights[:, :, :,  64: 96]
            g_RNN1_y2 = weights[:, :, :,  96: 128]

            g_RNN2_x1 = weights[:, :, :,  128: 160]
            g_RNN2_y1 = weights[:, :, :,  160: 192]
            g_RNN2_x2 = weights[:, :, :,  192: 224]
            g_RNN2_y2 = weights[:, :, :,  224: 256]

            rnn1_1 = lrnn(features, g_RNN1_x1, horizontal=True,  reverse=False)
            rnn1_2 = lrnn(features, g_RNN1_x2, horizontal=True,  reverse=True)
            rnn1_3 = lrnn(features, g_RNN1_y1, horizontal=False, reverse=False)
            rnn1_4 = lrnn(features, g_RNN1_y2, horizontal=False, reverse=True)
            rnn1 = tf.concat([rnn1_1, rnn1_2, rnn1_3, rnn1_4], axis=-1)
            conv_rnn1 = conv2d(rnn1, 32, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable) + features

            rnn2_1 = lrnn(conv_rnn1, g_RNN2_x1, horizontal=True,  reverse=False)
            rnn2_2 = lrnn(conv_rnn1, g_RNN2_x2, horizontal=True,  reverse=True)
            rnn2_3 = lrnn(conv_rnn1, g_RNN2_y1, horizontal=False, reverse=False)
            rnn2_4 = lrnn(conv_rnn1, g_RNN2_y2, horizontal=False, reverse=True)
            rnn2 = tf.concat([rnn2_1, rnn2_2, rnn2_3, rnn2_4], axis=-1)
            conv_rnn2 = conv2d(rnn2, 32, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable) + conv_rnn1

            attention_filtered = attention_module(conv_rnn2, tf.identity(features), tf.identity(conv_rnn2), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='attention_block')+conv_rnn2
            
            conv_restore = conv2d(attention_filtered, 16, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)
            inp_pred = conv2d(conv_restore, self.chns, self.cnn_size, activation=None, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable) + inputs
            var_reuse=True
        
        return inp_pred

    def build_model(self, mode='train'):
        self.gt_inp = tf.compat.v1.placeholder(shape=[None, None, None, self.chns], dtype=tf.float32) 
        self.gibbs_inp = tf.compat.v1.placeholder(shape=[None, None, None, self.chns], dtype=tf.float32) 
        self.is_training = tf.compat.v1.placeholder(shape=(), dtype=tf.bool) 

        self.cnn_out = self.generator(self.gibbs_inp, 'train', reuse=False, scope='g_net')
        
        self.loss_per_sample = tf.reduce_mean(
            tf.abs(self.gt_inp - self.cnn_out) + tf.compat.v1.losses.get_regularization_losses(), axis=[1,2,3])
        self.loss_total = tf.reduce_mean(tf.abs(self.gt_inp - self.cnn_out) + tf.compat.v1.losses.get_regularization_losses())
        #self.loss_total = tf.reduce_mean(tf.abs(self.gt_inp - self.cnn_out))
        
        self.rmse = tf.reduce_mean((self.gt_inp - self.cnn_out)**2, axis=[1,2,3])
        self.mae = tf.reduce_mean(tf.abs(self.gt_inp - self.cnn_out), axis=[1,2,3])
        self.psnr = tf.image.psnr(
                        tf.clip_by_value(self.gt_inp, 0., self.max_value), 
                        tf.clip_by_value(self.cnn_out, 0., self.max_value), 
                        max_val=self.max_value)

        all_vars = tf.compat.v1.trainable_variables()
        self.all_vars = all_vars
        for var in all_vars:
            print('Var name: {0}; Var shape: {1}'.format(var.name, var.get_shape()))

    def load_train(self):
        def train_generator_fn():
            i = 0
            while True:
                if i == len(self.data_list):
                    i = 0
                    random.shuffle(self.data_list)
                yield tuple(self.data_list[i])
                i = i + 1

        def augmentation(img, angle, s, rot, flip_params):
            if ((angle>0) or (s>1)):
                img = ScaleRotate(img, angle, s)
            if rot!=0:
                img = np.rot90(img, k=rot)
            if flip_params['doflip']:
                img = np.flip(img, axis=flip_params['flip_axis'])

            return img

        def numpy_read_image_fn(path_gt, path_sketch, angle, s, rot, doflip, flip_axis):
            img_gt = np.load(path_gt).astype(np.float32)
            img_sketch = np.load(path_sketch)

            params = {}
            params['sub'] = np.amin(img_gt)
            params['scale'] = np.amax(img_gt) - np.amin(img_gt)
            
            img_gt = data_normaliser(img_gt, mode=self.normalizer_data, value=params)
            img_sketch = data_normaliser(img_sketch, mode=self.normalizer_data, value=params)
            img_sketch = np.abs(img_sketch).astype(np.float32)

            flip_params = {'doflip':doflip, 'flip_axis':flip_axis}
            img_gt = augmentation(img_gt, angle, s, rot, flip_params)
            img_sketch = augmentation(img_sketch, angle, s, rot, flip_params)

            return img_gt, img_sketch

        def load_images(path_gt, path_sketch):
            # angle = np.random.uniform(-5.0, 5.0)
            # s = np.random.uniform(1.0, 1.5)
            angle = 0.
            s = 1.
            rot = np.random.randint(low=0, high=4) % 4
            doflip = bool(np.random.randint(low=0, high=2))
            flip_axis = np.random.randint(low=0, high=2) % 2
            
            img_gt, img_sketch = tf.numpy_function(numpy_read_image_fn, [tf.strings.join([self.data_root,'\\', path_gt]), tf.strings.join([self.data_root,'\\', path_sketch]), angle, s, rot, doflip, flip_axis], (tf.float32, tf.float32))

            return img_gt, img_sketch

        def get_patches(gt, sketch):
            patches = []
            cat = tf.concat([gt, sketch], -1)

            for i in range(self.num_random_crops):
                patch = tf.image.random_crop(cat, [self.crop_size, self.crop_size, self.chns + self.chns])
                patches.append(patch)
            patches = tf.stack(patches)
            assert patches.get_shape().dims == [self.num_random_crops, self.crop_size, self.crop_size, self.chns + self.chns]

            return patches

        dataset_train = (tf.data.Dataset.from_generator(train_generator_fn, (tf.string, tf.string))
                            .map(lambda path_gt, path_sketch: load_images(path_gt, path_sketch))
                            .map(lambda gt, sketch: get_patches(gt, sketch))
                            .apply(tf.contrib.data.unbatch())
                            .map(lambda x: tf.cast(x, tf.float32))
                            .shuffle(buffer_size=self.buffer_size)
                            .batch(batch_size=self.batch_size)
                            .prefetch(1)
                            .map(lambda HL: (HL[:, :, :, :self.chns], HL[:, :, :, self.chns:]))
                            )

        train_iterator = dataset_train.make_initializable_iterator()
        self.init_train_iterator = train_iterator.initializer
        self.next_batch_train = train_iterator.get_next()

    def train(self):
        self.global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)

        self.build_model()
        self.best_metrics = {'loss':-1., 'rmse':-1., 'mae':-1., 'psnr':-1.}
        self.load_train()

        # Learning rate decay.
        self.lr = tf.compat.v1.train.polynomial_decay(self.learning_rate, self.global_step, self.max_steps, end_learning_rate=0.0,
                                                      power=0.3)
        # Training operators.
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        train_gnet = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss_total, self.global_step, self.all_vars)
        train_gnet = tf.group([train_gnet, update_ops])

        # Session and thread.
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(self.init_train_iterator)
        self.saver = tf.compat.v1.train.Saver()

        # For possible transfer learning.
        train_meter = Meter()
        if (self.args.restore_ckpt > 0):
            ckpt_name = self.model_name + '-' + str(self.args.restore_ckpt)
            self.saver.restore(self.sess, os.path.join(self.train_dir, ckpt_name))
            df = pd.read_csv(os.path.join(self.train_dir, 'val.csv'))
            idx_best = df['PSNR'].idxmax()
            self.best_metrics['loss'] = df.iloc[idx_best]['Loss']
            self.best_metrics['rmse'] = df.iloc[idx_best]['RMSE']
            self.best_metrics['mae'] = df.iloc[idx_best]['MAE']
            self.best_metrics['psnr'] = df.iloc[idx_best]['PSNR']
            print('[*] Loaded checkpoint and current best metrics', ckpt_name)

        for step in range(self.sess.run(self.global_step), self.max_steps + 1):
            # self.data_size - how many batches we process per each epoch
            # step - per each step we process one batch
            epoch = step // self.data_size
            gt, sketch = self.sess.run(self.next_batch_train)
            
            feed_dict = {self.gt_inp: gt, self.gibbs_inp: sketch, self.is_training: True}
            start_time = time.time()
            _, loss_total, rmse_current, mae_current, psnr_current = self.sess.run([train_gnet, self.loss_total, self.rmse, self.mae, self.psnr], feed_dict=feed_dict)
            duration = time.time() - start_time
            rmse_current, mae_current, psnr_current = np.mean(rmse_current), np.mean(mae_current), np.mean(psnr_current)
            # assert not np.isnan(loss_total), 'Model diverged with loss = NaN'
            train_meter.update({'loss':loss_total, 'rmse':rmse_current, 'mae':mae_current, 'psnr':psnr_current})

            # Print loss info periodically.
            if step % 50 == 0:
                examples_per_sec = self.batch_size / (duration+1e-4)
                sec_per_batch = float(duration)

                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': Epoch {0}/{1}, Step {2}/{3}; Loss {4:.5f}, PSNR {5:.2f}, MAE {6:.5f}; ({7:.3f} data/s; {8:.3f} s/bch)'.format(
                    epoch,self.epoch,step-epoch*self.data_size,self.data_size,loss_total,psnr_current,mae_current,examples_per_sec,sec_per_batch))

            # Save the model checkpoint periodically.
            if (step % self.data_size == 0 or step == self.max_steps) and (step != 0):
                avg_meter = train_meter.average()
                WriteToCSVFile(os.path.join(self.train_dir, 'train.csv'), [[step, avg_meter['loss'], avg_meter['rmse'], avg_meter['mae'], avg_meter['psnr']]], mode='a')
                print('Epoch train statistics:\n  Loss = {0:.5f}\n  RMSE = {1:.5f}\n  MAE = {2:.5f}\n  PSNR = {3:.2f}'.format(avg_meter['loss'], avg_meter['rmse'], avg_meter['mae'], avg_meter['psnr']))
                print('Saving...')
                self.save(self.sess, self.train_dir, step)
                train_meter.reset()
                print('Validating...')
                self.validate(step)
        
    def save(self, sess, checkpoint_dir, step):
        epoch = step // self.data_size
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)
        if epoch > 0:
            prev_step = int((epoch-1)*self.data_size)
            for delfile in glob.glob(os.path.join(checkpoint_dir, self.model_name + '-' + str(prev_step) + '*')):
                if os.path.exists(delfile):
                    os.remove(delfile)

    def val_generator_fn(self, path_gt, path_sketch, batch_size):
        i = 0
        while True:
            sketch_batch = []
            gt_batch = []
            names_batch = []
            for b in range(batch_size):
                if i == len(path_sketch):
                    i = 0
                img_gt = np.load(path_gt[i]).astype(np.float32)
                img_sketch = np.load(path_sketch[i])
                
                params = {}
                params['sub'] = np.amin(img_gt)
                params['scale'] = np.amax(img_gt) - np.amin(img_gt)
                
                img_gt = data_normaliser(img_gt, mode=self.normalizer_data, value=params)
                img_sketch = data_normaliser(img_sketch, mode=self.normalizer_data, value=params)
                img_sketch = np.abs(img_sketch).astype(np.float32)

                sketch_batch.append(img_sketch)
                gt_batch.append(img_gt)
                names_batch.append(os.path.splitext(os.path.split(path_gt[i])[-1])[0])
                i = i + 1

            yield (np.array(sketch_batch), np.array(gt_batch), np.array(names_batch))

    def validate(self, step=None):
        def save_batch(gt_batch, sketch_batch, pred_batch, names_batch):
            error_batch = np.abs(pred_batch - gt_batch)
            for i in range(len(gt_batch)):
                sketch_batch[i] = data_normaliser(sketch_batch[i], mode='maxmin')
                pred_batch[i] = data_normaliser(pred_batch[i], mode='maxmin')
                gt_batch[i] = data_normaliser(gt_batch[i], mode='maxmin')
                error_batch[i] = data_normaliser(error_batch[i], mode='maxmin')
                imsave(os.path.join(self.val_path_gibbs_save, names_batch[i]+'_gibbs.png'), (sketch_batch[i]*255.).astype(np.uint8))
                imsave(os.path.join(self.val_path_degibbs_save, names_batch[i]+'_degibbs.png'), (pred_batch[i]*255.).astype(np.uint8))
                imsave(os.path.join(self.val_path_gt_save, names_batch[i]+'_gt.png'), (gt_batch[i]*255.).astype(np.uint8))
                imsave(os.path.join(self.val_path_error_save, names_batch[i]+'_error.png'), (error_batch[i]*255.).astype(np.uint8))

            return

        def visualize(amount=5):
            paths2vis = self.val_data_list[np.random.randint(low=0, high=len(self.val_data_list), size=amount)]
            path_list_gt = np.array([os.path.join(self.val_data_root, path) for path in paths2vis[:,0]])
            path_list_sketch = np.array([os.path.join(self.val_data_root, path) for path in paths2vis[:,1]])

            sketch_batch = []
            gt_batch = []
            error_batch = []
            for i in range(amount):
                img_gt = np.load(path_list_gt[i]).astype(np.float32)
                img_sketch = np.load(path_list_sketch[i])
                
                params = {}
                params['sub'] = np.amin(img_gt)
                params['scale'] = np.amax(img_gt) - np.amin(img_gt)
                
                img_gt = data_normaliser(img_gt, mode=self.normalizer_data, value=params)
                img_sketch = data_normaliser(img_sketch, mode=self.normalizer_data, value=params)
                img_sketch = np.abs(img_sketch).astype(np.float32)

                sketch_batch.append(img_sketch)
                gt_batch.append(img_gt)

            pred_batch = self.sess.run(self.cnn_out, feed_dict={self.gt_inp: gt_batch, self.gibbs_inp: sketch_batch, self.is_training: False})
            error_batch = np.abs(pred_batch - gt_batch)

            out = []
            for i in range(amount):
                sketch_batch[i] = data_normaliser(sketch_batch[i], mode='maxmin')
                pred_batch[i] = data_normaliser(pred_batch[i], mode='maxmin')
                gt_batch[i] = data_normaliser(gt_batch[i], mode='maxmin')
                error_batch[i] = data_normaliser(error_batch[i], mode='maxmin')
                out.append(np.hstack((sketch_batch[i], pred_batch[i], gt_batch[i], error_batch[i])))
            out = np.clip(np.vstack(out), 0., 1.)
            imsave(os.path.join(self.train_dir, 'comparison_{}.png'.format(step)), (out*255.).astype(np.uint8))
            return

        if self.args.phase == 'train':
            temp_val_data_list = self.val_data_list[np.random.randint(low=0, high=len(self.val_data_list), size=2000)].copy()
            path_list_gt = np.array([os.path.join(self.val_data_root, path) for path in temp_val_data_list[:,0]])
            path_list_sketch = np.array([os.path.join(self.val_data_root, path) for path in temp_val_data_list[:,1]])
            batch_size = 10
        else:
            temp_val_data_list = self.test_data_list.copy()
            path_list_gt = np.array([os.path.join(self.test_data_root, path) for path in temp_val_data_list[:,0]])
            path_list_sketch = np.array([os.path.join(self.test_data_root, path) for path in temp_val_data_list[:,1]])
            batch_size = 4

            self.build_model(mode='test')
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()
            ckpt_name = self.model_name + '-best'
            self.saver.restore(self.sess, os.path.join(self.train_dir, ckpt_name))

        val_meter = Meter()
        
        steps = len(path_list_sketch) // batch_size
        my_gen = self.val_generator_fn(path_list_gt.copy(), path_list_sketch.copy(), batch_size)
        with pb.ProgressBar(max_value = steps, widgets = ['Processing: ', pb.Percentage(), pb.Bar(), pb.ETA()]) as pbar:
            for i in range(steps):
                sketch_batch, gt_batch, names_batch = my_gen.__next__()
                loss_current, rmse_current, mae_current, psnr_current, pred_batch = self.sess.run([self.loss_per_sample, self.rmse, self.mae, self.psnr, self.cnn_out], feed_dict={self.gt_inp: gt_batch, self.gibbs_inp: sketch_batch, self.is_training: False})
                val_meter.update({'loss':loss_current, 'rmse':rmse_current, 'mae':mae_current, 'psnr':psnr_current}, step=len(psnr_current))
                if self.args.output_path != "":
                    save_batch(gt_batch, sketch_batch, pred_batch, names_batch)
                    for b in range(batch_size):
                        WriteToCSVFile(os.path.join(self.train_dir, 'test.csv'), [[names_batch[b], loss_current[b], rmse_current[b], mae_current[b], psnr_current[b]]], mode='a')
                pbar.update(i + 1)
        avg_meter = val_meter.average()
        if self.args.phase == 'test':
            print('Test average statistics:\n  Loss = {0:.5f}\n  RMSE = {1:.5f}\n  MAE = {2:.5f}\n  PSNR = {3:.2f}'.format(avg_meter['loss'], avg_meter['rmse'], avg_meter['mae'], avg_meter['psnr']))
            str2save = 'loss={0:.5f}\nrmse={1:.5f}\nmae={2:.5f}\npsnr={3:.2f}'.format(avg_meter['loss'], avg_meter['rmse'], avg_meter['mae'], avg_meter['psnr'])
            WriteToTXTFile(os.path.join(self.train_dir, 'test.txt'), str2save, mode='w')
        if self.args.phase == 'train':
            visualize(amount=4)
            WriteToCSVFile(os.path.join(self.train_dir, 'val.csv'), [[step, avg_meter['loss'], avg_meter['rmse'], avg_meter['mae'], avg_meter['psnr']]], mode='a')
            print('Epoch validation statistics:\n  Loss = {0:.5f}\n  RMSE = {1:.5f}\n  MAE = {2:.5f}\n  PSNR = {3:.2f}'.format(avg_meter['loss'], avg_meter['rmse'], avg_meter['mae'], avg_meter['psnr']))
            if (avg_meter['psnr'] >= self.best_metrics['psnr']):
                print('[*] New BEST!')
                self.best_metrics['loss'] = avg_meter['loss']
                self.best_metrics['rmse'] = avg_meter['rmse']
                self.best_metrics['mae'] = avg_meter['mae']
                self.best_metrics['psnr'] = avg_meter['psnr']
                str2save = 'step={0}\nloss={1:.5f}\nrmse={2:.5f}\nmae={3:.5f}\npsnr={4:.2f}'.format(step,self.best_metrics['loss'],self.best_metrics['rmse'],self.best_metrics['mae'],self.best_metrics['psnr'])
                WriteToTXTFile(os.path.join(self.train_dir, 'best.txt'), str2save, mode='w')
                for copyfile in glob.glob(os.path.join(self.train_dir, self.model_name + '-' + str(step) + '*')):
                    folder, name = os.path.split(copyfile)
                    _, ext = os.path.splitext(name) 
                    shutil.copyfile(copyfile, os.path.join(folder, self.model_name + '-best' + ext))
                shutil.copyfile(os.path.join(self.train_dir, 'comparison_{}.png'.format(step)), os.path.join(self.train_dir, 'comparison_best.png'))
        
        return


