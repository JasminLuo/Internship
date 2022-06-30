import numpy as np
import pandas as pd
import pmdarima as pm
import xgboost as xgb
from arch import arch_model
from PyEMD import CEEMDAN
from pyentrp import entropy as ent
from sklearn.model_selection import GridSearchCV, train_test_split
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


# Define ARIMA model fitting (without transformation)
class ARIMAWrapper(object):
    def __init__(self, data):
        self.data = data
    
    # Use auto_arima function to decide the best parameter of arima(p,d,q)
    def fit_arima(self,train_data):
        model = pm.arima.auto_arima(train_data, 
                                    information_criterion='aic',
                                    seasonal=False,   # No Seasonality
                                    error_action='ignore'
                                    )
        return model
    
    def fit_predict(self,split=10):
        # split data into train and test datasets
        # default to predict last 10 values of the time series
        train_y, test_y = train_test_split(self.data,test_size = split,shuffle = False)

        # do one-step forcasting
        forecast_data = pd.DataFrame()
        model = self.fit_arima(train_y)
        for t in range(len(test_y)):
            model_fit = model.fit(train_y)
            forecast = pd.DataFrame(model_fit.predict(n_periods=1))
            # forecast = pd.DataFrame(forecast, index=test_y[t:t + 1].index, columns=['Prediction'])
            # add the current step's forcasting data into dataframe
            forecast_data = pd.concat([forecast_data,forecast],axis=0)
            # add the real data into training set
            # retrain the arima model
            train_y = pd.concat([train_y,test_y[t:t + 1]],axis=0)
            model = self.fit_arima(train_y)
        forecast_data = forecast_data.to_numpy().ravel().tolist()
        # forecast_data = forecast_data.values.tolist()
        return forecast_data

# Define ARIMA model fitting (with CEEMDAN-PE transformation)
class CEEMDAN_PE_ARIMAWrapper(object):
    def __init__(self, data):
        self.data = data
        self.sum_imfs = None
        
    # Define CEEMDAN-PE transformation
    def ceemdan_pe_transf(self,train_data,order=6,delay=3):
        # generate IMFs from CEEMDAN decomposition
        ceemdan = CEEMDAN(trials = 500) #generate 500 times decompostion
        ceemdan.ceemdan(np.array(train_data).ravel())
        cimfs, cres = ceemdan.get_imfs_and_residue() # Extract cimfs and residue
        cimfs_df = pd.DataFrame(cimfs.T)

        # claculate PE value for each IMFs
        pe_imfs = []
        for i in range(cimfs_df.shape[1]):
            pe_imfs.append(ent.permutation_entropy(cimfs_df.values[:,i],order=order,delay=delay,normalize=True))
        
        # combination of IMFs with close PE values   
        for n in range(cimfs_df.shape[1]):
            pe_value = ent.permutation_entropy(cimfs_df.values[:,n],order=order,delay=delay,normalize=True)    
            
            if n==0:
                combined_series = pd.DataFrame(cimfs_df.iloc[:,n])
            elif pe_value_previous - pe_value <= 0.1:
                combined_series.iloc[:,-1] = combined_series.iloc[:,-1] + cimfs_df.iloc[:,n]
            else:            
                combined_series = pd.concat([combined_series,cimfs_df.iloc[:,n]],axis=1)
            pe_value_previous = pe_value
            
        combined_series = combined_series.reset_index(drop=True).T.reset_index(drop=True).T
        return combined_series

    def fit_arima(self,train_data):
        model = pm.arima.auto_arima(train_data, 
                                    information_criterion='aic',
                                    seasonal=False,   # No Seasonality
                                    error_action='ignore'
                                    )
        return model
    
    def fit_predict(self,split=10):
        # split data into train and test datasets
        # default to predict last 10 values of the time series
        train_y, test_y = train_test_split(self.data,test_size = split,shuffle = False)

        forecast_data_coms = pd.DataFrame()
        
        for t in range(len(test_y)):
            combined_series = self.ceemdan_pe_transf(train_y)
            # Define prediction data
            forecast_data = pd.DataFrame()

            # fit arima model for each combined time series
            for i in range(combined_series.shape[1]):
                com = combined_series[i]
                
                # fit model with auto ARIMA
                model = self.fit_arima(com)        
                model_fit = model.fit(com)
                # do one-step foracating for each combined time series
                forecast = pd.DataFrame(model_fit.predict(n_periods=1))
                forecast.columns = ['Combine Model %s' % i]
                forecast_data = pd.concat([forecast_data,forecast],axis=1)
            
            forecast_data_coms = pd.concat([forecast_data_coms,forecast_data],axis=0)
            # add the real data into training set
            train_y = pd.concat([train_y,test_y[t:t+1]],axis=0)
        # sum up the prediction value for each combined time series 
        sum_coms= forecast_data_coms.sum(axis=1)
        sum_coms = sum_coms.values.tolist()
        return sum_coms

# Define ARIMA - GARCH model fitting (without transformation)
class ARIMA_GARCHWrapper(object):
    def __init__(self, data):
        self.data = data
        
    def fit_arima(self,train_data):
        model = pm.arima.auto_arima(train_data, 
                                    information_criterion='aic',
#                                     m=12,              # frequency of series
                                    seasonal=False,   # No Seasonality
                                    error_action='ignore')
        return model
    
    # Find the best GARCH model from GARCH(1,1),GARCH(1,2),GARCH(2,1),GARCH(2,2)
    def best_garch_fit(self,arima_resid,disp):
        for i in [1,2]:
            for t in [1,2]:
                mdl_garch = arch_model(arima_resid, 
                                       vol = 'GARCH', 
                                       p = i, q = t,
                                       rescale = True
                                      )
                res_fit = mdl_garch.fit(disp=False)
                if i==1 & t==1:
                    best_p,best_q = i,t
                    min_aic = res_fit.aic
                # record the best parameters GARCH(p,q)
                best_p = int(np.where(res_fit.aic <= min_aic,i,best_p))
                best_q = int(np.where(res_fit.aic <= min_aic,i,best_q))
        garch_model = arch_model(arima_resid, vol = 'GARCH', p = best_p, q = best_q,rescale=True)
        garch_fit = garch_model.fit(disp=disp)
        return garch_fit
    
    def fit_predict(self,split=10):
        # split data into train and test datasets
        train_y, test_y = train_test_split(self.data,test_size = split,shuffle = False)

        model = self.fit_arima(train_y)
        forecast_data = pd.DataFrame()
        
        for t in range(len(test_y)):
            # Fit ARIMA Model
            model_fit = model.fit(train_y)
            # forecast_data  ARIMA
            forecast = model_fit.predict(n_periods=1)
            forecast = pd.DataFrame(forecast, index=test_y[t:t + 1].index, columns=[' Prediction'])

            # Test White noise for residual
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
                    # add GARCH component into forecast value
                    forecast = forecast + garch_pred 
            forecast_data = pd.concat([forecast_data,forecast],axis=0)
            train_y = pd.concat([train_y,test_y[t:t + 1]],axis=0)
            model = self.fit_arima(train_y)
        forecast_data = forecast_data.to_numpy().ravel().tolist()
        return forecast_data

# Define ARIMA - GARCH model fitting (with transformation)
class CEEMDAN_PE_ARIMA_GARCHWrapper(object):
    def __init__(self, data):
        self.data = data
        
    # Define CEEMDAN-PE transformation
    # Defualt setting PE with order=6, delay=3
    def ceemdan_pe_transf(self,train_data,order=6,delay=3):
        # generate IMFs from CEEMDAN decomposition
        ceemdan = CEEMDAN(trials = 500) #generate 500 times decompostion
        ceemdan.ceemdan(np.array(train_data).ravel())
        cimfs, cres = ceemdan.get_imfs_and_residue() # Extract cimfs and residue
        cimfs_df = pd.DataFrame(cimfs.T)

        # claculate PE value for each IMFs
        pe_imfs = []
        for i in range(cimfs_df.shape[1]):
            pe_imfs.append(ent.permutation_entropy(cimfs_df.values[:,i],order=order,delay=delay,normalize=True))
        
        # combination of IMFs with close PE values   
        # threshold = 2*(np.max(pe_imfs)-np.min(pe_imfs))/len(pe_imfs) 
        for n in range(cimfs_df.shape[1]):
            pe_value = ent.permutation_entropy(cimfs_df.values[:,n],order=order,delay=delay,normalize=True)    
            
            if n==0:
                combined_series = pd.DataFrame(cimfs_df.iloc[:,n])
            elif pe_value_previous - pe_value <= 0.1:
                combined_series.iloc[:,-1] = combined_series.iloc[:,-1] + cimfs_df.iloc[:,n]
            else:            
                combined_series = pd.concat([combined_series,cimfs_df.iloc[:,n]],axis=1)
            pe_value_previous = pe_value
            
        combined_series = combined_series.reset_index(drop=True).T.reset_index(drop=True).T
        return combined_series    

    def fit_arima(self,train_data):
        model = pm.arima.auto_arima(train_data, 
                                    information_criterion='aic',
#                                     m=12,              # frequency of series
                                    seasonal=False,   # No Seasonality
                                    error_action='ignore')
        return model
    
    # Find the best GARCH model from GARCH(1,1),GARCH(1,2),GARCH(2,1),GARCH(2,2)
    def best_garch_fit(self,arima_resid,disp):
        for i in [1,2]:
            for t in [1,2]:
                mdl_garch = arch_model(arima_resid, 
                                       vol = 'GARCH', 
                                       p = i, q = t,
                                       rescale = True
                                      )
                res_fit = mdl_garch.fit(disp=False)
                if i==1 & t==1:
                    best_p,best_q = i,t
                    min_aic = res_fit.aic
                best_p = int(np.where(res_fit.aic <= min_aic,i,best_p))
                best_q = int(np.where(res_fit.aic <= min_aic,i,best_q))
        garch_model = arch_model(arima_resid, vol = 'GARCH', p = best_p, q = best_q,rescale=True)
        garch_fit = garch_model.fit(disp=disp)
        return garch_fit
    
    def fit_predict(self,split=10):
        # split data into train and test datasets
        # default to predict last 10 values of the time series
        train_y, test_y = train_test_split(self.data,test_size = split,shuffle = False)
        
        forecast_data_coms = pd.DataFrame()
        
        for t in range(len(test_y)):
            combined_series = self.ceemdan_pe_transf(train_y)
            # Define prediction data
            forecast_data = pd.DataFrame()
            
            for i in range(combined_series.shape[1]):
                com = combined_series[i]
                   
                # Fit ARIMA Model
                model = self.fit_arima(com)
                model_fit = model.fit(com)
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
                forecast = pd.DataFrame(forecast, columns=['Combined model %s' % i])
                forecast_data = pd.concat([forecast_data,forecast],axis=1)
                    
            forecast_data_coms = pd.concat([forecast_data_coms,forecast_data],axis=0)
            # add the real data into training set
            train_y = pd.concat([train_y,test_y[t:t + 1]],axis=0)
        # sum up the prediction value for each combined time series 
        sum_coms= forecast_data_coms.sum(axis=1)
        sum_coms = sum_coms.values.tolist()
        return sum_coms 

# Define XGBoost model fitting (without transformation)
class XGBoostWrapper(object):
    def __init__(self, data):
        self.data = data
    
    # Input a time series data as full_data with format pd.series
    # Choose lag to generate features by shifting the original time series data  
    def generate_x_y(self,lag):
        X = self.data.to_numpy()
        # Gnerate features
        X_matrix = []
        y = []
        for i in range(len(X) - lag):
            sample =[]
            for n in range(lag):
                sample.append(X[i+n])

            X_matrix.append(sample)
            y.append(X[i+lag])
        
        # Define dataset 
        # X_data: features 
        # y_data: target
        X_data = np.array(X_matrix)
        y_data =  np.array(y)
        return X_data,y_data
    
    # Setup regressor 
    def best_params(self,train_x,train_y):
        xgb_model = xgb.XGBRegressor() 

        # performance a grid search
        # Use grid search find the best hyper-parameter which simulate the lowest MAE score
        #### Will change the validation method!!! ####
        tweaked_model = GridSearchCV(
            xgb_model,
            {
                'max_depth':[1,2,5,10,20],
                'n_estimators':[20,30,50,70,100],
                'learning_rate':[0.1,0.2,0.3,0.4,0.5]
            },   
            cv = 3,   # change to timeseries split!!
            n_jobs = 1,
            scoring = 'neg_median_absolute_error')

        # Fit and find the best model
        tweaked_model.fit(train_x,train_y)

        learning_rate = tweaked_model.best_params_['learning_rate']
        max_depth= tweaked_model.best_params_['max_depth']
        n_estimators= tweaked_model.best_params_['n_estimators']
        
        return learning_rate,max_depth,n_estimators

    def fit_xgboost(self,train_x,train_y):
        learning_rate, max_depth, n_estimators = self.best_params(train_x,train_y)

        model = xgb.XGBRegressor(
            learning_rate= learning_rate, 
            max_depth= max_depth, 
            n_estimators= n_estimators
            )
        return model
    
    # Fitting the best XGBoost model and do prediction (Horizon = 1)
    def fit_predict(self,lag=3,split=10):
        # Define features and target
        x_data = self.generate_x_y(lag)[0]
        y_data = self.generate_x_y(lag)[1]

        # split data into train and test datasets
        # default to predict last 10 values of the time series
        train_y, test_y = train_test_split(y_data,test_size = split,shuffle = False)
        train_x, test_x = train_test_split(x_data,test_size = split,shuffle = False)
        
        forecast_data = pd.DataFrame()
        # step1: fit the best XGBoost model
        model = self.fit_xgboost(train_x,train_y)
        
        for t in range(len(test_y)):
            model_fit = model.fit(train_x,train_y)
            # step2: do one-step prediction by using the last row of training features
            forecast = pd.DataFrame(model_fit.predict(train_x[-1:]))
            forecast_data = pd.concat([forecast_data,forecast],axis=0)
            # step3: update newest features and target value
            train_y = np.concatenate([train_y,test_y[t:t+1]],axis=0)
            train_x = np.concatenate([train_x,test_x[t:t+1]],axis=0)
            # step4: retrain the XGBoost model and get new hyperparameters
            model = self.fit_xgboost(train_x,train_y)
        forecast_data = forecast_data.to_numpy().ravel().tolist()
        return forecast_data

# Define XGBoost model fitting (with transformation)
class CEEMDAN_PE_XGBoostWrapper(object):
    def __init__(self, data):
        self.data = data
    
    # Define CEEMDAN-PE transformation
    def ceemdan_pe_transf(self,train_data,order=6,delay=3):
        # generate IMFs from CEEMDAN decomposition
        ceemdan = CEEMDAN(trials = 500) #generate 500 times decompostion
        ceemdan.ceemdan(np.array(train_data).ravel())
        cimfs, cres = ceemdan.get_imfs_and_residue() # Extract cimfs and residue
        cimfs_df = pd.DataFrame(cimfs.T)

        # claculate PE value for each IMFs
        pe_imfs = []
        for i in range(cimfs_df.shape[1]):
            pe_imfs.append(ent.permutation_entropy(cimfs_df.values[:,i],order=order,delay=delay,normalize=True))
        
        # combination of IMFs with close PE values   
        for n in range(cimfs_df.shape[1]):
            pe_value = ent.permutation_entropy(cimfs_df.values[:,n],order=order,delay=delay,normalize=True)    
            
            if n==0:
                combined_series = pd.DataFrame(cimfs_df.iloc[:,n])
            elif pe_value_previous - pe_value <= 0.1:
                combined_series.iloc[:,-1] = combined_series.iloc[:,-1] + cimfs_df.iloc[:,n]
            else:            
                combined_series = pd.concat([combined_series,cimfs_df.iloc[:,n]],axis=1)
            pe_value_previous = pe_value
            
        combined_series = combined_series.reset_index(drop=True).T.reset_index(drop=True).T
        return combined_series

    # Input a time series data as full_data with format pd.series
    # Choose lag to generate features by shifting the original time series data  
    def generate_x_y(self,lag):
        X = self.data.to_numpy()
        # Gnerate features
        X_matrix = []
        y = []
        for i in range(len(X) - lag):
            sample =[]
            for n in range(lag):
                sample.append(X[i+n])

            X_matrix.append(sample)
            y.append(X[i+lag])
        
        # Define dataset 
        # X_data: features 
        # y_data: target
        X_data = np.array(X_matrix)
        y_data =  np.array(y)
        return X_data,y_data
    
    # Setup regressor 
    def best_params(self,train_x,train_y):
        xgb_model = xgb.XGBRegressor() 

        # performance a grid search
        # Use grid search find the best hyper-parameter which simulate the lowest MAE score
        # #### Will change the validation method!!! ####
        tweaked_model = GridSearchCV(
            xgb_model,
            {
                'max_depth':[1,2,5,10,20],
                'n_estimators':[20,30,50,70,100],
                'learning_rate':[0.1,0.2,0.3,0.4,0.5]
            },   
            cv = 3,   #change to timeseries split!!
            n_jobs = 1,
            scoring = 'neg_median_absolute_error')

        # Fit and find the best model
        tweaked_model.fit(train_x,train_y)

        learning_rate = tweaked_model.best_params_['learning_rate']
        max_depth= tweaked_model.best_params_['max_depth']
        n_estimators= tweaked_model.best_params_['n_estimators']
        
        return learning_rate,max_depth,n_estimators

    def fit_xgboost(self,train_x,train_y):
        learning_rate, max_depth, n_estimators = self.best_params(train_x,train_y)

        model = xgb.XGBRegressor(
            learning_rate= learning_rate, 
            max_depth= max_depth, 
            n_estimators= n_estimators
            )
        return model
    
    # Fitting the best XGBoost model and do prediction (Horizon = 1)
    def fit_predict(self,lag=3,split=10):
        # Define features and target
        x_data = self.generate_x_y(lag)[0]
        y_data = self.generate_x_y(lag)[1]

        # split data into train and test datasets
        train_y, test_y = train_test_split(y_data,test_size = split,shuffle = False)
        train_x, test_x = train_test_split(x_data,test_size = split,shuffle = False)

        forecast_data_coms = pd.DataFrame()
        
        for t in range(len(test_y)):
            combined_series = self.ceemdan_pe_transf(train_y) # The number of rows is the same as the number of train_y.
            forecast_data = pd.DataFrame()
            # do prediction for each combined time series
            for i in range(combined_series.shape[1]):
                com = combined_series[i]
            
                # fit the best XGBoost model
                model = self.fit_xgboost(train_x,com) # target becomes the com data
                model_fit = model.fit(train_x,com)
                # do one-step prediction by using the last row of training features
                forecast = pd.DataFrame(model_fit.predict(train_x[-1:]))
                forecast.columns = ['Combine Model %s' % i]
                forecast_data = pd.concat([forecast_data,forecast],axis=1)
            
            forecast_data_coms = pd.concat([forecast_data_coms,forecast_data],axis=0)   
            # update newest features and target value after each prediction
            train_y = np.concatenate([train_y,test_y[t:t+1]],axis=0)
            train_x = np.concatenate([train_x,test_x[t:t+1]],axis=0)
        # sum up the prediction value for each combined time series 
        sum_coms= forecast_data_coms.sum(axis=1)
        sum_coms = sum_coms.values.tolist()
        return sum_coms
