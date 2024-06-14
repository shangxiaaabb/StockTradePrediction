# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from spektral.layers import GCNConv

seed = 100

def file_name(file_dir,file_type='.csv'):#默认为文件夹下的所有文件
    lst = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if(file_type == ''):
                lst.append(file)
            else:
                if os.path.splitext(file)[1] == str(file_type):#获取指定类型的文件名
                    lst.append(file)
    return lst

def normalize0(inputs):
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        if maks != 0:
            normalized.append(eq / maks)
        else:
            normalized.append(eq)
    return np.array(normalized)


def normalize1(inputs):
    normalized = []
    for eq in inputs:
        mean = np.mean(eq)
        std = np.std(eq)
        if std != 0:
            normalized.append((eq - mean) / std)
        else:
            normalized.append(eq)
    return np.array(normalized)

def normalize(inputs):
    normalized = []
    for eq in inputs:
        with np.errstate(invalid='ignore'):
            eq_log = [np.log(x) if i < 5 else x for i, x in enumerate(eq)]
            eq_log1 = np.nan_to_num(eq_log).tolist()
            normalized.append(eq_log1)
    return np.array(normalized)


def k_fold_split(inputs, targets, K):
    # make sure everything is seeded
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.permutation(seed)
    tf.random.set_seed(seed)

    ind = int(len(inputs) / K)
    inputsK = []
    targetsK = []

    for i in range(0, K - 1):
        inputsK.append(inputs[i * ind:(i + 1) * ind])
        targetsK.append(targets[i * ind:(i + 1) * ind])

    inputsK.append(inputs[(i + 1) * ind:])
    targetsK.append(targets[(i + 1) * ind:])

    return inputsK, targetsK


def merge_splits(inputs, targets, k, K):
    if k != 0:
        z = 0
        inputsTrain = inputs[z]
        targetsTrain = targets[z]
    else:
        z = 1
        inputsTrain = inputs[z]
        targetsTrain = targets[z]

    for i in range(z + 1, K):
        if i != k:
            inputsTrain = np.concatenate((inputsTrain, inputs[i]))
            targetsTrain = np.concatenate((targetsTrain, targets[i]))

    return inputsTrain, targetsTrain, inputs[k], targets[k]


def targets_to_list(targets):
    targetList = np.array(targets)

    return targetList


def build_model(input_shape):
    reg_const = 0.0001

    wav_input = layers.Input(shape=input_shape, name='wav_input')
    graph_input = layers.Input(shape=(input_shape[0], input_shape[0]), name='graph_input')
    graph_features = layers.Input(shape=(input_shape[0], 2), name='graph_features')

    conv1_new = wav_input
    # conv1_new = tf.keras.layers.Reshape((input_shape[0], 1))(wav_input)
    conv1_new = GCNConv(8, activation='leaky_relu', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))(
        [conv1_new, graph_input])
    conv1_new = GCNConv(8, activation='leaky_relu', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))(
        [conv1_new, graph_input])
    conv1_new = GCNConv(8, activation='leaky_relu', use_bias=True, kernel_regularizer=regularizers.l2(reg_const))(
        [conv1_new, graph_input])

    last_column = conv1_new[:, :, -1]
    conv1_new = tf.expand_dims(last_column, axis=-1)
    print(conv1_new.shape)
    print(conv1_new)

    conv1_new = layers.Flatten()(conv1_new)
    conv1_new = layers.Dropout(0.3, seed=seed)(conv1_new)
    print(conv1_new.shape)
    print(conv1_new)

    merged = layers.Dense(16)(conv1_new)
    y_hat = layers.Dense(1)(merged)

    final_model = models.Model(inputs=[wav_input, graph_input, graph_features], outputs=y_hat)
    rmsprop = optimizers.legacy.RMSprop(learning_rate=0.00015, rho=0.9, epsilon=None, decay=0.)
    final_model.compile(optimizer=rmsprop, loss='mse')

    return final_model