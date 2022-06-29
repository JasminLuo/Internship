import pandas as pd
# In order to use CEEMDAN transformation method,import PyEMD
from PyEMD import CEEMDAN
import numpy as np
from pandas import Series
# from pandas_datareader import data
import numpy as np
from PyEMD import EMD, Visualisation
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
import inspect
import random
import tqdm
import orbit
from orbit.utils.dataset import load_m3monthly
from orbit.models import ETS
from orbit.diagnostics.plot import plot_predicted_data
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from pandas_datareader import data

import time

n_sample=10
data = load_m3monthly()
unique_keys = data['key'].unique().tolist()
if n_sample > 0 and n_sample < len(unique_keys):
    sample_keys = random.sample(unique_keys, n_sample)
#     just get the first 5 series for demo
    data = data[data['key'].isin(sample_keys)].reset_index(drop=True)
else:
    sample_keys = unique_keys

# Set col according to the original dataset
key_col='key'
response_col='value'
date_col='date'

class ARIMAWrapper(object):
    def __init__(self, data):
        self.data = data
        
    def fit_arima(self,data):
        model = pm.arima.auto_arima(data, 
                                    information_criterion='aic',
#                                     test='adf',  # use adftest to find optimal 'd'
#                                     m=12,              # frequency of series
                                    seasonal=False,   # No Seasonality
                                    error_action='ignore')
        return model
    
    def fit_predict(self):
        # split data into train and test datasets
        train_y = self.data[:int(0.8*(len(self.data)))]
        test_y = self.data[int(0.8*(len(self.data))):]
#         train_y = self.data[:-8]
#         test_y = self.data[-8:]
        
        # 
        forecast_data = pd.DataFrame()
        model = self.fit_arima(train_y)
        for t in range(len(test_y)):
            model_fit = model.fit(train_y)
            forecast = model_fit.predict(n_periods=1)
            forecast = pd.DataFrame(forecast, index=test_y[t:t + 1].index, columns=['Prediction'])
            #
            forecast_data = pd.concat([forecast_data,forecast],axis=0)
            #
            train_y = pd.concat([train_y,test_y[t:t + 1]],axis=0)
            model = self.fit_arima(train_y)
        return forecast_data

class CEEMDAN_ARIMAWrapper(object):
    def __init__(self, data):
        self.data = data
        self.sum_imfs = None
        
    def ceemdan_decomp(self,train_data):
        ceemdan = CEEMDAN()
        ceemdan.ceemdan(np.array(train_data).ravel())
        cimfs, cres = ceemdan.get_imfs_and_residue() # Extract cimfs and residue
        cimfs_df = pd.DataFrame(cimfs.T)
        return cimfs_df

    def fit_arima(self,train_data):
        model = pm.arima.auto_arima(train_data, 
                                    information_criterion='aic',
#                                     test='adf',  # use adftest to find optimal 'd'
#                                     m=12,              # frequency of series
                                    seasonal=False,   # No Seasonality
                                    error_action='ignore')
        return model
    
    def fit_predict(self):
        # split data into train and test datasets
        train_y = self.data[:int(0.8*(len(self.data)))]
        test_y = self.data[int(0.8*(len(self.data))):]
#         train_y = self.data[:-8]
#         test_y = self.data[-8:]
        forecast_data_imfs = pd.DataFrame()
        
        for t in range(len(test_y)):
            cimfs_df = self.ceemdan_decomp(train_y)
            # Define prediction data
            forecast_data = pd.DataFrame()

            for i in range(cimfs_df.shape[1]):
                imf = cimfs_df[i]
                imf.index = self.data[:len(train_y)].index
                
                # fit model with auto ARIMA
                model = self.fit_arima(imf)        
                model_fit = model.fit(imf)
                forecast = model_fit.predict(n_periods=1)
                forecast = pd.DataFrame(forecast, index=test_y[t:t+1].index, columns=['IMF%s' % i + ' Prediction'])
                forecast_data = pd.concat([forecast_data,forecast],axis=1)
                
            forecast_data_imfs = pd.concat([forecast_data_imfs,forecast_data],axis=0)
            train_y = pd.concat([train_y,test_y[t:t+1]],axis=0)
        sum_imfs= forecast_data_imfs.sum(axis=1)
        sum_imfs = pd.DataFrame(sum_imfs,columns=['Prediction'])
        return sum_imfs

class ARIMA_GARCHWrapper(object):
    def __init__(self, data):
        self.data = data
        self.sum_imfs = None
        
    def fit_arima(self,train_data):
        model = pm.arima.auto_arima(train_data, 
                                    information_criterion='aic',
#                                     test='adf',  # use adftest to find optimal 'd'
#                                     m=12,              # frequency of series
                                    seasonal=False,   # No Seasonality
                                    error_action='ignore')
        return model
    
    def best_garch_fit(self,arima_resid,disp):
        for i in [1,2]:
            for t in [1,2]:
                mdl_garch = arch_model(arima_resid, 
#                                        mean = 'AR',
                                       vol = 'GARCH', 
                                       p = i, q = t,
                                       rescale = True
                                      )
                res_fit = mdl_garch.fit(disp=False)
        #         res_fit.summary() # whether coefficient is significant
                if i==1 & t==1:
                    best_p,best_q = i,t
                    min_aic = res_fit.aic
                best_p = int(np.where(res_fit.aic <= min_aic,i,best_p))
                best_q = int(np.where(res_fit.aic <= min_aic,i,best_q))
        garch_model = arch_model(arima_resid, vol = 'GARCH', p = best_p, q = best_q,rescale=True)
        garch_fit = garch_model.fit(disp=disp)
        return garch_fit
    
    def fit_predict(self):
        # split data into train and test datasets
        train_y = self.data[:int(0.8*(len(self.data)))]
        test_y = self.data[int(0.8*(len(self.data))):]
#         train_y = self.data[:-8]
#         test_y = self.data[-8:]
        model = self.fit_arima(train_y)
        forecast_data = pd.DataFrame()
        
        for t in range(len(test_y)):
            # Fit ARIMA Model
            model_fit = model.fit(train_y)
            # forecast_data  ARIMA
            forecast = model_fit.predict(n_periods=1)
            forecast = pd.DataFrame(forecast, index=test_y[t:t + 1].index, columns=[' Prediction'])

            train_y = pd.concat([train_y,test_y[t:t + 1]],axis=0)

            # Test White noise
            # By Ljung-Box test(standardised residual)
            arima_resid = model.arima_res_.resid
            lb_test = acorr_ljungbox(arima_resid, lags = [10],return_df=True)
            lb_test = lb_test['lb_pvalue'].values[0]  # H0: White noise
            if lb_test > 0.05:
                # ARCH Effect Test
                # LM - test
                LM_test = het_arch(arima_resid, ddof = 4)[1]  # H0: Without ARCH Effect
                if LM_test < 0.05:
                    # Fit GARCH Model
                    # Assume the total number of residual is x, start training by using x-len(valid_y) number of exist residual
                    garch_fit = self.best_garch_fit(arima_resid,disp=False)
                    prediction = garch_fit.forecast(horizon=1,reindex=True)
                    garch_pred = prediction.mean.values[-1,:][0]
                    forecast = forecast + garch_pred 
            forecast_data = pd.concat([forecast_data,forecast],axis=0)
            model = self.fit_arima(train_y)
        return forecast_data

class CEEMDAN_ARIMA_GARCHWrapper(object):
    def __init__(self, data):
        self.data = data
        self.sum_imfs = None
        
    def ceemdan_decomp(self,train_data):
        ceemdan = CEEMDAN()
        ceemdan.ceemdan(np.array(train_data).ravel())
        cimfs, cres = ceemdan.get_imfs_and_residue() # Extract cimfs and residue
        cimfs_df = pd.DataFrame(cimfs.T)
        return cimfs_df    

    def fit_arima(self,train_data):
        model = pm.arima.auto_arima(train_data, 
                                    information_criterion='aic',
#                                     test='adf',  # use adftest to find optimal 'd'
#                                     m=12,              # frequency of series
                                    seasonal=False,   # No Seasonality
                                    error_action='ignore')
        return model
    
    def best_garch_fit(self,arima_resid,disp):
        for i in [1,2]:
            for t in [1,2]:
                mdl_garch = arch_model(arima_resid, 
                                       vol = 'GARCH', 
                                       p = i, q = t,
                                       rescale = True
                                      )
                res_fit = mdl_garch.fit(disp=False)
        #         res_fit.summary() # whether coefficient is significant
                if i==1 & t==1:
                    best_p,best_q = i,t
                    min_aic = res_fit.aic
                best_p = int(np.where(res_fit.aic <= min_aic,i,best_p))
                best_q = int(np.where(res_fit.aic <= min_aic,i,best_q))
        garch_model = arch_model(arima_resid, vol = 'GARCH', p = best_p, q = best_q,rescale=True)
        garch_fit = garch_model.fit(disp=disp)
        return garch_fit
    
    def fit_predict(self):
        # split data into train and test datasets
        train_y = self.data[:int(0.8*(len(self.data)))]
        test_y = self.data[int(0.8*(len(self.data))):]
#         train_y = self.data[:-8]
#         test_y = self.data[-8:]
        
        forecast_data_imfs = pd.DataFrame()
        
        for t in range(len(test_y)):
            cimfs_df = self.ceemdan_decomp(train_y)
            # Define prediction data
            forecast_data = pd.DataFrame()
            
            for i in range(cimfs_df.shape[1]):
                imf = cimfs_df[i]
                imf.index = self.data[:len(train_y)].index
                   
                # Fit ARIMA Model
                model = self.fit_arima(imf)
                model_fit = model.fit(imf)
                # forecast_data  ARIMA
                forecast = model_fit.predict(n_periods=1)
                
                # Test White noise
                # By Ljung-Box test(standardised residual)
                arima_resid = model.arima_res_.resid
                lb_test = acorr_ljungbox(arima_resid, lags = [10],return_df=True)
                lb_test = lb_test['lb_pvalue'].values[0]  # H0: White noise
                if lb_test > 0.05:
                    # ARCH Effect Test
                    # LM - test
                    LM_test = het_arch(arima_resid, ddof = 4)[1]  # H0: Without ARCH Effect
                    if LM_test < 0.05:
                        # Fit GARCH Model
                        # Assume the total number of residual is x, start training by using x-len(valid_y) number of exist residual
                        garch_fit = self.best_garch_fit(arima_resid,disp=False)
                        prediction = garch_fit.forecast(horizon=1,reindex=True)
                        garch_pred = prediction.mean.values[-1,:][0]
                        forecast = forecast + garch_pred
                forecast = pd.DataFrame(forecast, index=test_y[t:t + 1].index, columns=['IMF%s' % i + ' Prediction'])
                forecast_data = pd.concat([forecast_data,forecast],axis=1)
                    
            forecast_data_imfs = pd.concat([forecast_data_imfs,forecast_data],axis=0)
            train_y = pd.concat([train_y,test_y[t:t + 1]],axis=0)
        sum_imfs= forecast_data_imfs.sum(axis=1)
        sum_imfs = pd.DataFrame(sum_imfs,columns=['Prediction'])
        return sum_imfs 

def evaluate(data,forecast_data,index):
        a = np.array(data[int(0.8*(len(data))):])
#         a = np.array(self.data[-8:])
        f = np.array(forecast_data)
        
        rms = np.sqrt(mean_squared_error(a, f))
        smape = np.mean(2 * np.abs(f-a) / (np.abs(a) + np.abs(f)))
        scores = pd.DataFrame({'RMSE':[rms],'SMAPE':[smape]},index=[index])
        return scores
    
def save_json(info_dict):
    with open('./results.json.temp','a') as file:
        file.write(json.dumps(info_dict))
        file.write('\n')
        
def confirm_json():
    target_path = f'./result_{int(time.time())}.json'
    os.rename('./results.json.temp',target_path)

for key in tqdm.tqdm(sample_keys):
    # Initial new dict
    info_dict = {
        'dataset_name' : key,
        'lag': 0,
        'arima': dict(),
        'ceemdan_arima': dict()
    }
    
#     all_scores = pd.DataFrame()
    
    # Data precessing
    df = data[data[key_col] == key]
    df_copy = df.copy()
    infer_freq = pd.infer_freq(df_copy[date_col])
    df_copy = df_copy.set_index(date_col)
    df_copy = df_copy.asfreq(infer_freq).drop(['key'], axis=1)
    df_copy = df_copy['value']

    # Fit model
    arima = ARIMAWrapper(df_copy).fit_predict()
    ceemdan_arima = CEEMDAN_ARIMAWrapper(df_copy).fit_predict()
#     arima_garch = ARIMA_GARCHWrapper(df_copy).fit_predict()
#     ceemdan_arima_garch = CEEMDAN_ARIMA_GARCHWrapper(df_copy).fit_predict()
#     garch = GARCHWrapper(df_copy)
    
    info_dict['arima'] = {
        'pred' : arima
    }
    info_dict['ceemdan_arima'] = {
        'pred' : ceemdan_arima
    }
    
    save_json(info_dict)

confirm_json()


# Calculate scores
all_scores = pd.concat([all_scores,evaluate(df_copy,arima,'ARIMA')],axis=0)
all_scores = pd.concat([all_scores,evaluate(df_copy,ceemdan_arima,index='CEEMDAN - ARIMA')],axis=0)
all_scores = pd.concat([all_scores,evaluate(df_copy,arima_garch,index='ARIMA - GARCH')],axis=0)
all_scores = pd.concat([all_scores,evaluate(df_copy,ceemdan_arima_garch,index='CEEMDAN - ARIMA - GARCH')],axis=0)
# all_scores = pd.concat([all_scores,garch.evaluate()],axis=0)

plot the predictions for validation set
plt.figure(figsize=(8,5))
plt.plot(df_copy[:int(0.8*(len(data)))], label='Train')
plt.plot(df_copy[int(0.8*(len(df_copy))):], label='Test')
plt.plot(arima, label='ARIMA Prediction')
plt.plot(ceemdan_arima,label = 'CEEMDAN - ARIMA Prediction')
plt.plot(arima_garch,label = 'ARIMA - GARCH Prediction')
plt.plot(ceemdan_arima_garch,label = 'CEEMDAN - ARIMA - GARCH Prediction')
plt.legend(loc = 'upper left')
plt