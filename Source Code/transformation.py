from PyEMD import EMD
import numpy as np
import pandas as pd

# EMD-based transformation
def emd_tranf(train_data):
    # generate IMFs from EMD decomposition
    emd = EMD() 
    emd.emd(np.array(train_data).ravel(),max_imf=4)
    imfs, res = emd.get_imfs_and_residue() # Extract cimfs and residue
    imfs = pd.DataFrame(imfs).T
    res = pd.DataFrame(res)
    imfs_df = pd.concat([imfs,res],axis=1)

    residual = imfs_df.iloc[:,1:].sum(axis=1)

    return np.array(residual).ravel()


# Detrend
def detrend(insample_data):
    """
    Calculates a & b parameters of LRL
    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b

def add_trend(ts,forecast,a,b,horizon=1):
    for i in range(0, horizon):
        forecast[i] = forecast[i] + ((a * (len(ts) + i + 1)) + b)
    return forecast


# Deseasonlisatioon
def acf(data, k):
    """
    Autocorrelation function
    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)

def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS
    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    if window % 2 == 0:
        ts_ma = ts_init.rolling(window, center=True).mean()
        ts_ma = ts_ma.rolling(2, center=True).mean()
        ts_ma = np.roll(ts_ma, -1)
    else:
        ts_ma = ts_init.rolling(window, center=True).mean()

    return ts_ma

def seasonality_test(original_ts, ppy):
    """
    Seasonality test
    :param original_ts: time series
    :param ppy: periods per year
    :return: boolean value: whether the TS is seasonal
    """
    s = acf(original_ts, 1)**2
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (np.sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit

def deseasonalize(original_ts, ppy):
    original_ts = pd.Series(original_ts)
    """
    Calculates and returns seasonal indices
    :param original_ts: original data
    :param ppy: periods per year
    :return:
    """
    if seasonality_test(original_ts, ppy):
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        si = np.full(ppy, 100)

    return si

def remove_seanson(ts,fh):
    seasonality_in = deseasonalize(ts, fh)

    for i in range(0, len(ts)):
        ts[i] = ts[i] * 100 / seasonality_in[i % fh]
    return ts

def add_season(ts,forecast,fh,horizon=1):
    seasonality_in = deseasonalize(ts, fh)

    for i in range(len(ts), len(ts) + horizon):
        forecast[i - len(ts)] = forecast[i - len(ts)] * seasonality_in[i % fh] / 100
    
    return forecast