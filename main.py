import random
import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import orbit
import pandas as pd
import pmdarima as pm
import tqdm
import xgboost as xgb
import os
import json
from arch import arch_model
from orbit.diagnostics.plot import plot_predicted_data
from orbit.models import ETS
from orbit.utils.dataset import load_m3monthly
from pandas import Series
from pandas_datareader import data
from PyEMD import CEEMDAN, EMD, Visualisation
from pyentrp import entropy as ent
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from Wrapper import ARIMAWrapper,CEEMDAN_PE_ARIMAWrapper,XGBoostWrapper,CEEMDAN_PE_XGBoostWrapper


def save_json(info_dict):
    with open('./results.json.temp','a') as file:
        file.write(json.dumps(info_dict))
        file.write('\n')

def confirm_json():
    target_path = f'./Target-Transformation/prediction_results/result_{int(time.time())}.json'
    print(target_path)
    os.rename('./results.json.temp',target_path)
def init_json():
    if os.path.exists('./results.json.temp'):
        os.remove('./results.json.temp')
def sample_keys(n_sample=1):
    ### M3 Dataset
    data = load_m3monthly()
    unique_keys = data['key'].unique().tolist()
    if n_sample > 0 and n_sample < len(unique_keys):
        sample_keys = random.sample(unique_keys, n_sample)
        # just get the first 5 series for demo
        data = data[data['key'].isin(sample_keys)].reset_index(drop=True)
    else:
        sample_keys = unique_keys
    return sample_keys

if __name__ == '__main__':
    init_json()
    
    for key in tqdm.tqdm(sample_keys()):
        # Initial new dict
        info_dict = {
            'dataset_name' : key,
            # 'lag': 3,
            'arima': dict(),
            'arima_transf': dict(),
            'xgboost': dict(),
            'xgboost_transf': dict()
        }

        
        # Set col according to the original dataset
        data = load_m3monthly()
        key_col='key'
        response_col='value'
        date_col='date'
        # Data precessing
        df = data[data[key_col] == key]
        df_copy = df.copy()
        infer_freq = pd.infer_freq(df_copy[date_col])
        df_copy = df_copy.set_index(date_col)
        df_copy = df_copy.asfreq(infer_freq).drop(['key'], axis=1)
        df_copy = df_copy['value']

        # Fit model
        arima = ARIMAWrapper(df_copy).fit_predict()
        # arima_transf = CEEMDAN_PE_ARIMAWrapper(df_copy).fit_predict()
        # xgboost = XGBoostWrapper(df_copy).fit_predict()
        # xgboost_transf = CEEMDAN_PE_XGBoostWrapper(df_copy).fit_predict()

        # Save forecast value into info_dict 
        info_dict['arima'] = {
            'pred' : arima
        # }
        # info_dict['arima_tranf'] = {
        #     'pred' : arima_transf
        # }
        # info_dict['xgboost'] = {
        #     'pred' : xgboost
        # }
        # info_dict['xgboost_transf'] = {
        #     'pred' : xgboost_transf
        }

        save_json(info_dict)
    confirm_json()
