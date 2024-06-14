# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import random
from GCN_new import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

if __name__ == "__main__":
    # stock_info = sys.argv[0]
    # lag_bin = int(sys.argv[1])
    # lag_day = int(sys.argv[2])
    # bin_num = int(sys.argv[3])
    # random_state_here = int(sys.argv[4])
    # test_set_size = float(sys.argv[5])
    lag_bin = 3
    lag_day = 3
    num_nodes = (int(lag_bin)+1)*(int(lag_day)+1)
    bin_num = 24
    forecast_days = 14
    random_state_here = 88
    mape_list = []
    data_dir = './data/volume/0308/'
    files =file_name('./data/')
    stocks_info = list(set(s.split('_25')[0] for s in files))
    print(stocks_info)
    for stock_info in stocks_info:
        print(f'>>>>>>>>>>>>>>>>>>>>{stock_info}>>>>>>>>>>>>>>>>>>>>>>>')
        print(f'>>>>>>>>>>>>>>>>>>>>{stock_info}>>>>>>>>>>>>>>>>>>>>>>>')
        data_dir1 = f'{data_dir}{stock_info}_{lag_bin}_{lag_day}'
        test_set_size = bin_num*forecast_days
        K = 5
        inputs_data = np.load(f'{data_dir1}_inputs.npy', allow_pickle=True)#.astype(np.float64)
        inputs_data = [[[float(x) for x in sublist] for sublist in list1] for list1 in inputs_data]
        array_data = np.array(inputs_data)
        inputs = np.reshape(array_data, (len(inputs_data), num_nodes,-1))
        targets = np.load(f'{data_dir1}_output.npy', allow_pickle=True).astype(np.float64)
        graph_input = np.load(f'{data_dir1}_graph_input.npy', allow_pickle=True).astype(np.float64)
        graph_input = np.array([graph_input] * inputs.shape[0])

        graph_features = np.load(f'{data_dir1}_graph_coords.npy', allow_pickle=True).astype(np.float64)
        graph_features = np.array([graph_features] * inputs.shape[0])

        train_inputs, test_inputs, traingraphinput, testgraphinput, train_graphfeature, test_graphfeature, train_targets, testTargets = train_test_split(inputs, graph_input, graph_features, targets, test_size=test_set_size, 
                                                     random_state=random_state_here)
        testInputs = normalize(test_inputs)
        

        # testInputs = test_inputs
        inputsK, targetsK = k_fold_split(train_inputs, train_targets, K)

        mape_list = []
        for k in range(0,K):

            keras.backend.clear_session()
            tf.keras.backend.clear_session()

            trainInputsAll, trainTargets, valInputsAll, valTargets = merge_splits(inputsK, targetsK, k, K)

            train_graphinput = traingraphinput[0:trainInputsAll.shape[0], :]
            train_graphfeatureinput = train_graphfeature[0:trainInputsAll.shape[0], :]

            val_graphinput = traingraphinput[0:valInputsAll.shape[0], :]
            val_graphfeatureinput = train_graphfeature[0:valInputsAll.shape[0], :]

            trainInputs = normalize(trainInputsAll[:, :])
            valInputs = normalize(valInputsAll[:, :])
            
            trainInputs = trainInputsAll
            valInputs = valInputsAll
            model = build_model(valInputs[0].shape)

            es = keras.callbacks.EarlyStopping(patience=50, verbose=0, min_delta=0.001, monitor='val_loss', mode='min',
                                               baseline=None, restore_best_weights=True)

            iteration_checkpoint = keras.callbacks.ModelCheckpoint(
                f'models/{stock_info}_{lag_bin}_{lag_day}_gcn_model_iteration_{k}.h5',
                monitor='val_loss',
                verbose=0,
                save_best_only=True
            )

            print(model.summary())

            history = model.fit(x=[trainInputs, train_graphinput, train_graphfeatureinput], y=trainTargets, epochs=5000,
                                batch_size=50,
                                validation_data=([valInputs, val_graphinput, val_graphfeatureinput], valTargets),
                                verbose=0, callbacks=[es,iteration_checkpoint])  #

            print()
            print('total number of epochs ran = ', len(history.history['loss']))
            print('Fold number:' + str(k))
            predictions = model.predict([testInputs, testgraphinput, test_graphfeature])
            # predictions = model.predict([trainInputs, train_graphinput, train_graphfeatureinput])

            new_predictions = np.array(predictions)
            new_predictions = [item for sublist in new_predictions for item in sublist]
            new_predictions = [testTargets[i-1] if np.isnan(new_predictions[i]) else new_predictions[i] for i in range(len(new_predictions))]

            MAPE = []
        
            # MAPE.append(mean_absolute_percentage_error(trainTargets[:], new_predictions[:]))
            # testTargets0 = [item for sublist in trainTargets for item in sublist]

            MAPE.append(mean_absolute_percentage_error(testTargets[:], new_predictions[:]))
            print(MAPE)
            # testTargets0 = [item for sublist in testTargets for item in sublist]
            testTargets0 = list(testTargets)

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
