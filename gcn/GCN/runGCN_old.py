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
from sklearn.model_selection import train_test_split
from spektral.layers import GCNConv
from sklearn.metrics import mean_absolute_percentage_error
import genGCNdata0
import matplotlib.pyplot as plt
import sys
# %matplotlib inline
# Set device to CPU or GPU
# device = 'CPU'
device = 'GPU'  # Uncomment this line and comment the previous line to use GPU

if device == 'GPU':
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

seed = 1234


def normalize0(inputs):
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        if maks != 0:
            normalized.append(eq / maks)
        else:
            normalized.append(eq)
    return np.array(normalized)


def normalize(inputs):
    normalized = []
    for eq in inputs:
        eq_log = np.log(eq)
        normalized.append(eq_log)
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

    conv1_new = tf.keras.layers.Reshape((input_shape[0], 1))(wav_input)
    conv1_new = GCNConv(8, activation='leaky_relu', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))(
        [conv1_new, graph_input])
    conv1_new = GCNConv(8, activation='leaky_relu', use_bias=False, kernel_regularizer=regularizers.l2(reg_const))(
        [conv1_new, graph_input])
    conv1_new = GCNConv(8, activation='leaky_relu', use_bias=True, kernel_regularizer=regularizers.l2(reg_const))(
        [conv1_new, graph_input])

    last_column = conv1_new[:, :, -1]
    conv1_new = tf.expand_dims(last_column, axis=-1)

    conv1_new = layers.Flatten()(conv1_new)
    conv1_new = layers.Dropout(0.3, seed=seed)(conv1_new)

    merged = layers.Dense(16)(conv1_new)
    y_hat = layers.Dense(1)(merged)

    final_model = models.Model(inputs=[wav_input, graph_input, graph_features], outputs=y_hat)
    rmsprop = optimizers.RMSprop(learning_rate=0.00015, rho=0.9, epsilon=None, decay=0.)
    final_model.compile(optimizer=rmsprop, loss='mse')

    return final_model


if __name__ == "__main__":
    lag_bin = 3
    lag_day = 3
    bin_num = 25
    random_state_here = 1234
    mape_list = []
    num_epochs = 1000
    batch_size = 50

    # stocks_info =["000049_XSHE","300006_XSHE","000048_XSHE","002046_XSHE",
    #               "002651_XSHE","000004_XSHE","300180_XSHE","002780_XSHE","300023_XSHE","000005_XSHE",
    #               "000407_XSHE","300162_XSHE","300717_XSHE","000026_XSHE","000009_XSHE"]
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
    data_dir = './data/'
    files =file_name(data_dir)
    stocks_info = list(set(s.split('_25')[0] for s in files))
    print(stocks_info)

    for stock_info in stocks_info:
        print(f'>>>>>>>>>>>>>>>>>>>>{stock_info}>>>>>>>>>>>>>>>>>>>>>>>')
        genGCNdata0.genLeftUpAllData(lag_bin, lag_day, bin_num, stock_info)
        test_set_size = 14*25
        K = 5
        inputs = np.load(f'data/{stock_info}_{lag_bin}_{lag_day}_inputs.npy', allow_pickle=True).astype(np.float64)
        targets = np.load(f'data/{stock_info}_{lag_bin}_{lag_day}_output.npy', allow_pickle=True).astype(np.float64)
        graph_input = np.load(f'data/{stock_info}_{lag_bin}_{lag_day}_graph_input.npy', allow_pickle=True).astype(
            np.float64)
        graph_input = np.array([graph_input] * inputs.shape[0])

        graph_features = np.load(f'data/{stock_info}_{lag_bin}_{lag_day}_graph_coords.npy', allow_pickle=True).astype(
            np.float64)
        graph_features = np.array([graph_features] * inputs.shape[0])

        train_inputs, test_inputs, traingraphinput, testgraphinput, train_graphfeature, test_graphfeature, train_targets, testTargets = train_test_split(
            inputs, graph_input, graph_features, targets, test_size=test_set_size, random_state=random_state_here)
        testInputs = normalize(test_inputs)
        inputsK, targetsK = k_fold_split(train_inputs, train_targets, K)

        mape_list = []
        for k in range(4, K):
            keras.backend.clear_session()
            tf.keras.backend.clear_session()

            trainInputsAll, trainTargets, valInputsAll, valTargets = merge_splits(inputsK, targetsK, k, K)

            train_graphinput = traingraphinput[0:trainInputsAll.shape[0], :]
            train_graphfeatureinput = train_graphfeature[0:trainInputsAll.shape[0], :]

            val_graphinput = traingraphinput[0:valInputsAll.shape[0], :]
            val_graphfeatureinput = train_graphfeature[0:valInputsAll.shape[0], :]

            trainInputs = normalize(trainInputsAll[:, :])
            valInputs = normalize(valInputsAll[:, :])

            model = build_model(valInputs[0].shape)

            es = keras.callbacks.EarlyStopping(patience=50, verbose=0, min_delta=0.005, monitor='val_loss', mode='min',
                                               baseline=None, restore_best_weights=True)

            iteration_checkpoint = keras.callbacks.ModelCheckpoint(
                f'models/{stock_info}_{lag_bin}_{lag_day}_gcn_model_iteration_{k}.h5',
                monitor='val_loss',
                verbose=0,
                save_best_only=True
            )

            print(model.summary())

            history = model.fit(x=[trainInputs, train_graphinput, train_graphfeatureinput], y=trainTargets, epochs=num_epochs,
                                batch_size=batch_size,
                                validation_data=([valInputs, val_graphinput, val_graphfeatureinput], valTargets),
                                verbose=0, callbacks=[es, iteration_checkpoint])  

            print()
            print('total number of epochs ran = ', len(history.history['loss']))
            print('Fold number:' + str(k))
            predictions = model.predict([testInputs, testgraphinput, test_graphfeature])

            new_predictions = np.array(predictions)
            new_predictions = [item for sublist in new_predictions for item in sublist]
            MAPE = []
            MAPE.append(mean_absolute_percentage_error(testTargets[:], new_predictions[:]))
            
            testTargets0 = [item for sublist in testTargets for item in sublist]

            res = {
                'testTargets': testTargets0,
                'new_predictions': new_predictions
            }

            res_df = pd.DataFrame(res)
            res_df.to_csv(f'./result/{stock_info}_{lag_bin}_{lag_day}_res_test_MAPE{k}.csv', index=False)
            
           

            print('MAPE = ', np.array(MAPE).mean())
            MAPE_mean = np.array(MAPE).mean()
            mape_list.append(MAPE)

            keras.backend.clear_session()
            tf.keras.backend.clear_session()

        print('-')
        print('mape score = ', mape_list)

        # Plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{stock_info}_Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(f'./result/{stock_info}_loss_plot.png')  # 保存损失图像
        plt.show()

        # Plot predictions
        plt.plot(testTargets0)
        plt.plot(new_predictions)
        plt.title(f'{stock_info}_Predictions vs Actual')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend(['Actual', 'Predictions'], loc='upper right')
        plt.savefig(f'./result/{stock_info}_test_predictions_plot.png')  # 保存损失图像
        plt.show()
