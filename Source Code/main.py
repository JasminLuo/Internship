import json
import logging
import multiprocessing as mp
import os
import time
import warnings
from multiprocessing import pool

import numpy as np
from orbit.utils.dataset import load_m3monthly
from sklearn.model_selection import train_test_split

from Models import Model

warnings.filterwarnings('ignore')
logger1 = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s %(levelname)-7s] %(message)s', datefmt='%Y%m%d %H:%M:%S')
logger1.setLevel(logging.INFO)

def save_json(info):
    with open('./data.json.temp','a') as file:
        file.write(json.dumps(info))
        file.write('\n')

def confirm_json():
    path = f'./Target-Transformation/prediction_results/results_{int(time.time())}.json'
    print(path)
    os.rename('./data.json.temp',path)

def init_json():
    if os.path.exists('./data.json.temp'):
        os.remove('./data.json.temp')

def load_m3_data(min_length=100, n_set=1500):
    datasets = dict()
    m3_data = load_m3monthly()
    keys = m3_data['key'].unique().tolist()

    print('Loading M3 dataset')
    for key in keys:
        current = m3_data[m3_data['key'] == key]
        if current.shape[0] > min_length:
            datasets[key] = current['value'].to_numpy()
            if len(datasets.keys()) == n_set:
                break
    return datasets

def fit_predict(data):
    # Transformation Methods
    detrend_deseason = True
    split = 0.2 
    fh = 12
    horizon = 1
    lags = 12

    key,series = data
    copy = np.array(series)
    logger1.info(key)

    # save train and test data
    train,test = train_test_split(copy,test_size=int(len(copy)*split),shuffle=False)

    # Select model wnat to use
    train_model = {
        'ARIMA': Model.ARIMA,
        'AutoETS': Model.AutoETS,
        'ElasticNet': Model.RegressionModel,
        'LinearSVR': Model.RegressionModel,
        'KNeighborsRegressor': Model.RegressionModel,
        'DecisionTreeRegressor': Model.RegressionModel,
        'MLPRegressor': Model.RegressionModel,
    }

    info_dict = {
        'method_info': {
            'detrend_deseason': detrend_deseason,
        },

        'dataset_info': {
            'dataset_name' : key,
            'origin_ts': copy.tolist(),
            "length": len(copy),
            'train_size': len(train),
            'test_size':len(test),
        },

        'train_test': {
            'train': train.tolist(),
            'test': test.tolist(),
        },

        'best_params': {
            'untransformed': {
                'ARIMA': [],
                'AutoETS': [],
                'ElasticNet': [],
                'LinearSVR': [],
                'KNeighborsRegressor': [],
                'DecisionTreeRegressor': [],
                'MLPRegressor': [],
            },
            'transformed': {
                'ARIMA': [],
                'AutoETS': [],
                'ElasticNet': [],
                'LinearSVR': [],
                'KNeighborsRegressor': [],
                'DecisionTreeRegressor': [],
                'MLPRegressor': [],
            },
        },

        'untransformed':{
            'ARIMA': [],
            'AutoETS': [],
            'ElasticNet': [],
            'LinearSVR': [],
            'KNeighborsRegressor': [],
            'DecisionTreeRegressor': [],
            'MLPRegressor': [],
        },

        'transformed':{
            'ARIMA': [],
            'AutoETS': [],
            'ElasticNet': [],
            'LinearSVR': [],
            'KNeighborsRegressor': [],
            'DecisionTreeRegressor': [],
            'MLPRegressor': [],
        },
    }


    # Fit model
    for method in ['untransformed','transformed']:
        if method == 'untransformed':
            for run_times in range(len(test)):
                # generate current train and test
                if run_times == 0:
                    train_y = np.copy(train)
                    test_y = np.copy(test)
                else:
                    train_y = np.concatenate([train_y,test_y[run_times-1:run_times]],axis=0)

                for model in train_model.keys():
                    # print(info_dict['best_params'][model])
                    model_info = Model(train = train_y, real_test = test_y, fh = fh, \
                        horizon = horizon, lags = lags, \
                        detrend = detrend_deseason, method = method, model = model, \
                        run_time = run_times, best_parameter = info_dict['best_params'][method][model])
                    
                    if run_times == 0 and model not in ['ARIMA','AutoETS']:
                        pred, best_param = train_model[model](model_info)
                        info_dict['best_params'][method][model] = best_param
                    else:
                        pred = train_model[model](model_info)

                    info_dict[method][model].append(pred)

        if method == 'transformed':
            for run_times in range(len(test)):
                # generate current train and test
                if run_times == 0:
                    train_y = np.copy(train)
                    test_y = np.copy(test)
                else:
                    train_y = np.concatenate([train_y,test_y[run_times-1:run_times]],axis=0)


                for model in train_model.keys():
                    model_info = Model(train = train_y, real_test = test_y, fh = fh, \
                        horizon = horizon, lags = lags, \
                        detrend = detrend_deseason, method = method, model = model, \
                        run_time = run_times, best_parameter = info_dict['best_params'][method][model])
                    
                    if run_times == 0 and model not in ['ARIMA','AutoETS']:
                        pred, best_param = train_model[model](model_info)
                        info_dict['best_params'][method][model] = best_param
                    else:
                        pred = train_model[model](model_info)

                    info_dict[method][model].append(pred)

    save_json(info_dict)


if __name__ == '__main__':
    n_set = 1
    min_length = 100
    # worker = 1

    init_json()
    data = load_m3_data(min_length=min_length,n_set=n_set)

    # Multi Processing
    # pool = mp.Pool(worker)
    

    var = [key for key in data.items()]
    for v in var:
        fit_predict(v)
    # pool.map(fit_predict,var)
    # pool.close()
    # pool.join()
    confirm_json()
