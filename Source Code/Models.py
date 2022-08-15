import numpy as np
import pandas as pd
from sklearn import tree, linear_model, neighbors, neural_network, svm
from sklearn.model_selection import ParameterGrid
from sktime.forecasting import ets, statsforecast

import transformation as tf



class Model:
    def __init__(self, train, real_test, fh=12, horizon=1, lags=12, 
    detrend = True, method='untransformed',model='ElasticNet', run_time=1, best_parameter=[]):
        '''
        train: Input training data
        fh: periods per year
        horizon: prediction steps, how many data want to predict at once
        lags: generate feature based on this
        '''

        self.train = train
        self.real_test = real_test
        self.fh = fh
        self.horizon = horizon
        self.lags = lags
        self.detrend = detrend
        self.method = method
        self.model = model
        self.run_time = run_time
        self.best_params = best_parameter
        
        ## parameters setting
        self.params = {
            'ElasticNet': {
                'alpha': [0.01, 0.1, 1, 10],
                'l1_ratio': np.arange(0.1, 1., 0.2),
                'tol': [0.001],
                'selection': ['random'],
            },

            'LinearSVR': {
                'epsilon': np.arange(0., 1., 0.4),
                'C': [1, 5, 10, 20],
            },

            'KNeighborsRegressor': {
                'n_neighbors': [2, 4, 8, 12, 16, 20],
                'algorithm': ['auto'],
            },

            'DecisionTreeRegressor': {
                'ccp_alpha': np.arange(0, 1, 0.4),
            },

            'MLPRegressor': {
                'hidden_layer_sizes': [(10, 10, 10,), (10, 10), (20, 10), (10,), (20,), ],
                'shuffle': [False],
                'max_iter': [500],
                'learning_rate_init': [0.1],
                'learning_rate': ['adaptive'],
            },
        }
        
        ## reegression models
        self.regression = {
            'ElasticNet': linear_model.ElasticNet,
            'LinearSVR': svm.LinearSVR,
            'KNeighborsRegressor': neighbors.KNeighborsRegressor,
            'DecisionTreeRegressor': tree.DecisionTreeRegressor,
            'MLPRegressor': neural_network.MLPRegressor,
            }

    ########################################
    ## Validation - Choose the Best Model ##
    ########################################
    def transform(self,train_):
        train = np.copy(train_)
        if self.detrend:

            train_y = tf.remove_seanson(train, fh=self.fh)
            a, b = tf.detrend(train_y)

            for i in range(0, len(train_y)):
                train_y[i] = train_y[i] - ((a * i) + b)

        else:
            a, b = None, None            
            train_y = train

        if self.method == 'transformed':
            train_trans = tf.emd_tranf(train_y)
        else:
            train_trans = train_y

        return a, b, train_trans, train_y

    def detrend_add(self, pred, train_y, a=None, b=None):
        if self.detrend:
            pred = np.array(pred).ravel()
            # add trend and seasonal
            pred = tf.add_trend(train_y, pred, a, b, horizon = self.horizon)
            pred = tf.add_season(train_y, pred, fh=self.fh)

        if isinstance(pred, np.ndarray):
            return np.array(pred[0])
        
        return np.array(pred)

    def gen_features(self,train):
        '''
        Generate features for regression models
        '''
        data = []
        data_trans = []

        a, b, train_trans,train_y = self.transform(train)
        # exit()
        # to generate feature, add one dummy variable at the end of the list
        train = list(train)
        train_trans = list(train_trans)
        train.append(0)
        train_trans.append(0)

        for i in range(len(train) - self.horizon - self.lags + 1):
            index = i + self.horizon + self.lags
            
            data.append(train[i:index])
            data_trans.append(train_trans[i:index])

        X_t = np.array(data_trans)[:,:self.lags]
        y_t = np.array(data_trans)[:,self.lags:]
        return X_t, y_t, a, b, train_y

    def expand_window(self,param):
        test_len = len(self.real_test)
        pred = []

        for i in range(test_len):
            valid_y = self.train[:-test_len+i]

            X_t, y_t, a, b,train_y = self.gen_features(valid_y)
            
            regression_model = self.regression[self.model]()
            regression_model.set_params(**param)
            regression_model.fit(X_t[:-1], y_t[:-1])
            prediction = regression_model.predict(X_t[-1:])[0]

            prediction = self.detrend_add(prediction,train_y,a,b)

            pred.append(prediction)
 
        return pred

    # Gride Search the best parameters
    def get_best_param(self):
        best_score = np.inf

        params = list(ParameterGrid(self.params[self.model]))

        for param in params:
            pred = self.expand_window(param)

            a = np.array(self.train[-len(self.real_test):])
            f = pred
            
            # calculate SMAPE score
            smape_scores = np.mean(2 * np.abs(f-a) / (np.abs(a) + np.abs(f)))

            # update the best model
            if smape_scores < best_score:
                best_score = smape_scores
                best_param = param

        return best_param

    def best_model(self,params):
        X_t, y_t, _, _, _ = self.gen_features(self.train)

        best_model = self.regression[self.model]()
        best_model.set_params(**params)
        best_model.fit(X_t[:-1], y_t[:-1])
        
        return best_model


    #######################    
    ## forecasting model ##
    #######################
    def ARIMA(self):
        a_, b_, train_trans, train_y = self.transform(self.train)
        model = statsforecast.StatsForecastAutoARIMA()
        model_fit = model.fit(train_trans)
        prediction = model_fit.predict(fh=[self.horizon])
        prediction = self.detrend_add(prediction, train_y, a_, b_).tolist()
        return prediction

    def AutoETS(self):
        a_, b_, train_trans, train_y = self.transform(self.train)
        model = ets.AutoETS(auto=True)
        model.fit(train_trans)
        prediction = model.predict(fh=self.horizon)
        prediction = self.detrend_add(prediction, train_y, a_, b_).tolist()
        
        return prediction
    
    #######################    
    ## regression model ###
    #######################    
    def RegressionModel(self):
        a_, b_, train_trans, train_y = self.transform(self.train)
        X = train_trans[-self.lags:].reshape(-1,self.lags)

        # tune hyperparameters
        if self.run_time==0:
            best_param = self.get_best_param()
            model = self.best_model(best_param)
            prediction = model.predict(X)
            prediction = self.detrend_add(prediction,train_y,a_,b_)
            return prediction.item(), best_param

        else:
            model = self.best_model(self.best_params)
            prediction = model.predict(X)
            prediction = self.detrend_add(prediction,train_y,a_,b_)
            return prediction.item()




    





