import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import functools

from .functions import _Function
from joblib import wrap_non_picklable_objects
import pandas_ta as ta
import talib
from talib import MA_Type ## MA_Type.SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3
import bottleneck as bn
import numba as nb
import time
import polars as pl
import numbagg as nbgg



_extra_function_map= {}
tszs_wins = [60, 120]
ts_wins = [1, 3, 5, 10, 20, 40, 60]
ts_lags = [1, 3, 5, 10] # used in like autocorr, etc.


perf_data = {}


# region ==== Utils ====

def default_ts_zscore(x, nan_mask=np.array([]), w=2000):
    m = nbgg.move_mean(x, window=w, min_count=int(w/2), axis=0)
    s = nbgg.move_std(x, window=w, min_count=int(w/2), axis=0)
    z = np.divide(x - m, s, out=np.zeros_like(x), where=s!=0)
    z = bn.replace(z, -np.inf, 0)
    z = bn.replace(z, np.inf, 0)
    z = bn.replace(z, np.nan, 0)
    z = np.clip(z, -6, 6)
    if nan_mask.size > 0:
        z[nan_mask] = 0
    return z

def pre_and_post_process(func):

    @functools.wraps(func)
    def wrapper(A, *args, **kwargs):
    
        # 处理 NumPy 的错误状态
        with np.errstate(over='ignore', under='ignore'):
            t0 = time.perf_counter()

            ## save nan mask
            nan_mask = np.isnan(A)

            ## forward fill on col 
            A = nbgg.ffill(A, axis=0)

            ## call func
            result = func(A, *args, **kwargs)

            ## default ts zscore
            result = default_ts_zscore(result, nan_mask)
            
            ## time logging
            n = func.__name__
            t = time.perf_counter() - t0
            global perf_data
            if n in perf_data:
                perf_data[n]['count'] += 1
                perf_data[n]['ttl_t'] += t
                perf_data[n]['avg_t'] = perf_data[n]['ttl_t'] / perf_data[n]['count']
            else:
                perf_data[n] = {'count': 1, 'ttl_t': t, 'avg_t': t}

        ## check shape
        assert(result.shape == A.shape)

        return result
    
    return wrapper


def pre_and_post_process_2(func):

    @functools.wraps(func)
    def wrapper(A, B, *args, **kwargs):
    
        # 处理 NumPy 的错误状态
        with np.errstate(over='ignore', under='ignore'):
            t0 = time.perf_counter()

            ## save nan mask
            nan_mask = np.isnan(A) 

            ## forward fill on col 
            A = nbgg.ffill(A, axis=0)
            B = nbgg.ffill(B, axis=0)    

            ## call func
            result = func(A, B, *args, **kwargs)

            ## default ts zscore
            result = default_ts_zscore(result, nan_mask)
            
            ## time logging
            n = func.__name__
            t = time.perf_counter() - t0
            global perf_data
            if n in perf_data:
                perf_data[n]['count'] += 1
                perf_data[n]['ttl_t'] += t
                perf_data[n]['avg_t'] = perf_data[n]['ttl_t'] / perf_data[n]['count']
            else:
                perf_data[n] = {'count': 1, 'ttl_t': t, 'avg_t': t}

        ## check shape
        assert(result.shape == A.shape)

        return result
    
    return wrapper


def pre_and_post_process_3(func):

    @functools.wraps(func)
    def wrapper(A, B, C, *args, **kwargs):
    
        # 处理 NumPy 的错误状态
        with np.errstate(over='ignore', under='ignore'):
            t0 = time.perf_counter()

            ## save nan mask
            nan_mask = np.isnan(A) 

            ## forward fill on col 
            A = nbgg.ffill(A, axis=0)
            B = nbgg.ffill(B, axis=0)  
            C = nbgg.ffill(C, axis=0)  

            ## call func
            result = func(A, B, C, *args, **kwargs)

            ## default ts zscore
            result = default_ts_zscore(result, nan_mask)
            
            ## time logging
            n = func.__name__
            t = time.perf_counter() - t0
            global perf_data
            if n in perf_data:
                perf_data[n]['count'] += 1
                perf_data[n]['ttl_t'] += t
                perf_data[n]['avg_t'] = perf_data[n]['ttl_t'] / perf_data[n]['count']
            else:
                perf_data[n] = {'count': 1, 'ttl_t': t, 'avg_t': t}

        ## check shape
        assert(result.shape == A.shape)

        return result
    
    return wrapper


def pre_and_post_process_4(func):

    @functools.wraps(func)
    def wrapper(A, B, C, D, *args, **kwargs):
    
        # 处理 NumPy 的错误状态
        with np.errstate(over='ignore', under='ignore'):
            t0 = time.perf_counter()

            ## save nan mask
            nan_mask = np.isnan(A) 

            ## forward fill on col 
            A = nbgg.ffill(A, axis=0)
            B = nbgg.ffill(B, axis=0)  
            C = nbgg.ffill(C, axis=0)  
            D = nbgg.ffill(D, axis=0)  

            ## call func
            result = func(A, B, C, D, *args, **kwargs)

            ## default ts zscore
            result = default_ts_zscore(result, nan_mask)
            
            ## time logging
            n = func.__name__
            t = time.perf_counter() - t0
            global perf_data
            if n in perf_data:
                perf_data[n]['count'] += 1
                perf_data[n]['ttl_t'] += t
                perf_data[n]['avg_t'] = perf_data[n]['ttl_t'] / perf_data[n]['count']
            else:
                perf_data[n] = {'count': 1, 'ttl_t': t, 'avg_t': t}

        ## check shape
        assert(result.shape == A.shape)

        return result
    
    return wrapper

## multiple arrays version of np.apply_along_axis()
def apply_column_2(x1, x2, func, *args, **kwargs):
    r = np.full_like(x1, np.nan)
    for i in range(x1.shape[1]):
        r[:, i] = func(x1[:, i], x2[:, i], *args, **kwargs)
    return r

## quite faster than pd.rolling.apply() but slower than pd.rolling.pd_support_func()
def np_rolling_apply(x, w, func):
    if w > x.shape[0]:
        return np.full_like(x, np.nan)

    roll_view = sliding_window_view(x, window_shape=w, axis=0)  # axis=0 split windows vertically (ts) 

    result = np.full_like(x, np.nan)

    for i in range(w - 1, x.shape[0]):
        result[i] = func(roll_view[i - w + 1], axis=1)  # apply func on each window horizontally. note: each window is a row in the roll_view!

    return result

# endregion


# region ==== Time Series functions ====


@pre_and_post_process
def ts_zscore(x, w=60):
    m = nbgg.move_mean(x, window=w, min_count=int(w/2), axis=0)
    s = nbgg.move_std(x, window=w, min_count=int(w/2), axis=0)
    z = (x - m) / s
    z[s == 0] = np.nan
    z = np.clip(z, -6, 6)
    return z
_extra_function_map.update({f'tszs_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_zscore(x, w)), name=f'tszs_{w}', arity=1) for w in tszs_wins})

@pre_and_post_process
def ts_delay(x, w=3):
    d = np.roll(x, w, axis=0)
    d[:w, :] = np.nan
    return d
_extra_function_map.update({f'ts_delay_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_delay(x, w)), name=f'ts_delay_{w}', arity=1) for w in ts_wins})

@pre_and_post_process
def ts_diff(x, w=3):    # numpy.diff is recursive, not this purpose
    d = np.roll(x, w, axis=0)
    d[:w, :] = np.nan
    d = x - d
    return d
_extra_function_map.update({f'ts_diff_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_diff(x, w)), name=f'ts_diff_{w}', arity=1) for w in ts_wins})

@pre_and_post_process
def ts_diff2nd(x, w=3):    # numpy.diff is recursive, not this purpose
    d1 = ts_diff(x, w)
    d2 = ts_diff(d1, w)
    return d2
_extra_function_map.update({f'ts_diff2nd_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_diff2nd(x, w)), name=f'ts_diff2nd_{w}', arity=1) for w in ts_wins})

@pre_and_post_process
def ts_roc(x, w=3): # rate of change;
    shifted_arr = np.roll(x, w, axis=0)
    shifted_arr[:w, :] = np.nan
    roc = (x - shifted_arr) / np.abs(shifted_arr) - 1
    roc[np.abs(shifted_arr) == 0] = np.nan
    return roc
_extra_function_map.update({f'ts_roc_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_roc(x, w)), name=f'ts_roc_{w}', arity=1) for w in ts_wins})

@pre_and_post_process
def ts_sum(x, w=3):
    # return bn.move_sum(x, window=w, min_count=int(w/2), axis=0)
    return nbgg.move_sum(x, window=w, min_count=int(w/2), axis=0)
_extra_function_map.update({f'ts_sum_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_sum(x, w)), name=f'ts_sum_{w}', arity=1) for w in ts_wins if w > 1})

@pre_and_post_process
def ts_mean(x, w=3):
    # return bn.move_mean(x, window=w, min_count=int(w/2), axis=0)
    return nbgg.move_mean(x, window=w, min_count=int(w/2), axis=0)
_extra_function_map.update({f'ts_mean_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_mean(x, w)), name=f'ts_mean_{w}', arity=1) for w in ts_wins if w > 1})

@pre_and_post_process
def ts_median(x, w=3):
    return bn.move_median(x, window=w, min_count=int(w/2), axis=0)
_extra_function_map.update({f'ts_median_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_median(x, w)), name=f'ts_median_{w}', arity=1) for w in ts_wins if w > 1})

@pre_and_post_process
def ts_std(x, w=5):
    return bn.move_std(x, window=w, min_count=int(w/2), axis=0, ddof=1)
    # return nbgg.move_std(x, window=w, min_count=int(w/2), axis=0)
_extra_function_map.update({f'ts_std_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_std(x, w)), name=f'ts_std_{w}', arity=1) for w in ts_wins if w >= 5})

@pre_and_post_process
def ts_skew(x, w=10):    # pd is best for now
    return pd.DataFrame(x).rolling(w, min_periods=int(w/2)).skew().to_numpy(dtype=np.double)
    # return pl.DataFrame(x, orient='row', nan_to_null=True).with_columns(pl.all().rolling_skew(w)).to_numpy()
_extra_function_map.update({f'ts_skew_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_skew(x, w)), name=f'ts_skew_{w}', arity=1) for w in ts_wins if w >= 10})

@pre_and_post_process
def ts_kurt(x, w=10):    # pd is best for now
    return pd.DataFrame(x).rolling(w, min_periods=int(w/2)).kurt().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_kurt_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_kurt(x, w)), name=f'ts_kurt_{w}', arity=1) for w in ts_wins if w >= 10})

@pre_and_post_process
def ts_max(x, w=5):
    return bn.move_max(x, window=w, min_count=int(w/2), axis=0)
_extra_function_map.update({f'ts_max_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_max(x, w)), name=f'ts_max_{w}', arity=1) for w in ts_wins if w > 1})

@pre_and_post_process
def ts_min(x, w=5):
    return bn.move_min(x, window=w, min_count=int(w/2), axis=0)
_extra_function_map.update({f'ts_min_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_min(x, w)), name=f'ts_min_{w}', arity=1) for w in ts_wins if w > 1})

@pre_and_post_process
def ts_rank(x, w=3):    # range [-1, 1], different with pd
    return bn.move_rank(x, window=w, min_count=int(w/2), axis=0)
_extra_function_map.update({f'ts_rank_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_rank(x, w)), name=f'ts_rank_{w}', arity=1) for w in ts_wins if w >= 3})

@pre_and_post_process
def ts_autocorr(x, w=5, l=1):   # no faster way for now
    """moving autocorrelation coefficient between x and x lag i period"""
    x_lagged = np.roll(x, l, axis=0)
    x_lagged[:l, :] = np.nan
    d = nbgg.move_corr(x, x_lagged, window=w, min_count=int(w/2), axis=0)
    return d
## too small window will not be stable
_extra_function_map.update({f'ts_autocorr_{w}_{l}': _Function(function=wrap_non_picklable_objects(lambda x, w=w, l=l: ts_autocorr(x, w, l)), name=f'ts_autocorr_{w}_{l}', arity=1) for w in ts_wins for l in ts_lags if w >=10 and l <= w})

@pre_and_post_process
def ts_argmin(x, w=3):
    return w - 1 - bn.move_argmin(x, window=w, min_count=int(w/2), axis=0) # move_argmin count index 0 from rightmost edge 
_extra_function_map.update({f'ts_argmin_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_argmin(x, w)), name=f'ts_argmin_{w}', arity=1) for w in ts_wins if w >=3})

@pre_and_post_process
def ts_argmax(x, w=3):
    return w - 1 - bn.move_argmax(x, window=w, min_count=int(w/2), axis=0)  # move_argmax count index 0 from rightmost edge
_extra_function_map.update({f'ts_argmax_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_argmax(x, w)), name=f'ts_argmax_{w}', arity=1) for w in ts_wins if w >=3})

@pre_and_post_process
def ts_quantile(x, w=5, q=0.25):
    return pl.DataFrame(x, orient='row', nan_to_null=True).with_columns(
        pl.all().rolling_quantile(q, window_size=w, min_periods=int(w/2), interpolation='linear')
    ).to_numpy()
_extra_function_map.update({f'ts_quantile_{w}_0.25': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_quantile(x, w, 0.25)), name=f'ts_quantile_{w}_0.25', arity=1) for w in ts_wins if w >=5})
_extra_function_map.update({f'ts_quantile_{w}_0.75': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_quantile(x, w, 0.75)), name=f'ts_quantile_{w}_0.75', arity=1) for w in ts_wins if w >=5})

@pre_and_post_process
def ts_decreasing(x, w=5):    # no faster way for now
    return (pd.DataFrame(x).diff(w) < 0).astype('int').to_numpy(np.double)
_extra_function_map.update({f'ts_decreasing_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_decreasing(x, w)), name=f'ts_decreasing_{w}', arity=1) for w in ts_wins if w >=3})

@pre_and_post_process
def ts_increasing(x, w=5):      # no faster way for now  
    return (pd.DataFrame(x).diff(w) > 0).astype('int').to_numpy(np.double)
_extra_function_map.update({f'ts_increasing_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_increasing(x, w)), name=f'ts_increasing_{w}', arity=1) for w in ts_wins if w >=3})

@pre_and_post_process
def ts_relative(x, w=5):
    sma = ta_SMA(x, timeperiod=w)
    sma = np.where(np.isnan(sma), 1, sma)
    return np.divide(x, sma, out=np.zeros_like(x), where=sma!=0)
_extra_function_map.update({f'ts_relative_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_relative(x, w)), name=f'ts_relative_{w}', arity=1) for w in ts_wins if w >=3})

## ER (Efficiency Ratio, https://mp.weixin.qq.com/s/wqr-o5CfksS_K3OO8PzK8g)
@pre_and_post_process
def ts_ER(x, w=5):
    # abs diff of delay w
    d = np.roll(x, w, axis=0)
    d[:w, :] = np.nan
    d = np.abs(x - d)
    # sum of abs of delay 1
    d2 = np.roll(x, 1, axis=0)
    d2[:w, :] = np.nan
    d2 = np.abs(x - d2)
    d2 = nbgg.move_sum(d2, window=w, min_count=int(w/2), axis=0)
    # er
    er = np.divide(d, d2, out=np.full(x.shape, np.nan), where=d2!=0)
    er = np.where((er < 0.0) | (er > 1.0), np.nan, er)  # range [0, 1]
    return er
_extra_function_map.update({f'ts_ER_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_ER(x, w)), name=f'ts_ER_{w}', arity=1) for w in ts_wins if w >=5})


#endregion


# region ==== Cross Section functions ====

@pre_and_post_process
def cs_rank(x):
    return bn.nanrankdata(x, axis=1) / bn.nansum(~np.isnan(x), axis=1)[:, np.newaxis]  # actually no much perf gain
_extra_function_map.update({'cs_rank': _Function(function=wrap_non_picklable_objects(lambda x: cs_rank(x)), name = 'cs_rank', arity=1)})

@pre_and_post_process
def cs_zscore(x):
    m = nbgg.nanmean(x, axis=1)[:, np.newaxis]
    s = nbgg.nanstd(x, axis=1, ddof=0)[:, np.newaxis]
    z = (x - m) / s
    z = np.clip(z, -6, 6)
    return z
_extra_function_map.update({'cs_zscore': _Function(function=wrap_non_picklable_objects(lambda x: cs_zscore(x)), name = 'cs_zscore', arity=1)})


# endregion


# region ==== TA functions ====

# region ==== TA functions: Momentum ====

@pre_and_post_process
def ta_APO(x, fastperiod, slowperiod, matype):
    return np.apply_along_axis(talib.APO, 0, x, fastperiod, slowperiod, matype)
_extra_function_map.update({f'ta_APO_12_26': _Function(function=wrap_non_picklable_objects(lambda x: ta_APO(x, 12, 26, 0)), name = 'ta_APO_12_26', arity=1)})

@pre_and_post_process
def ta_CMO(x, timeperiod):
    return np.apply_along_axis(talib.CMO, 0, x, timeperiod=timeperiod)
_extra_function_map.update({f'ta_CMO_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_CMO(x, w)), name=f'ta_CMO_{w}', arity=1) for w in [10]})

def _macd(x, fastperiod, slowperiod, signalperiod, type=''):
    m, s, h = talib.MACD(x, fastperiod, slowperiod, signalperiod)
    if type == 's':
        return s
    elif type == 'h':
        return h
    else:
        return m
@pre_and_post_process
def ta_MACD(x, fastperiod, slowperiod, signalperiod):
    return np.apply_along_axis(_macd, 0, x, fastperiod, slowperiod, signalperiod, '')
@pre_and_post_process
def ta_MACDs(x, fastperiod, slowperiod, signalperiod):
    return np.apply_along_axis(_macd, 0, x, fastperiod, slowperiod, signalperiod, 's')
@pre_and_post_process
def ta_MACDh(x, fastperiod, slowperiod, signalperiod):
    return np.apply_along_axis(_macd, 0, x, fastperiod, slowperiod, signalperiod, 'h')
_extra_function_map.update({
    'ta_MACD_12_26_9': _Function(function=wrap_non_picklable_objects(lambda x: ta_MACD(x, 12, 26, 9)), name = 'ta_MACD_12_26_9', arity=1),
    'ta_MACDs_12_26_9': _Function(function=wrap_non_picklable_objects(lambda x: ta_MACDs(x, 12, 26, 9)), name = 'ta_MACDs_12_26_9', arity=1),
    'ta_MACDh_12_26_9': _Function(function=wrap_non_picklable_objects(lambda x: ta_MACDh(x, 12, 26, 9)), name = 'ta_MACDh_12_26_9', arity=1),
})

@pre_and_post_process
def ta_PPO(x, fastperiod, slowperiod, matype):
    return np.apply_along_axis(talib.PPO, 0, x, fastperiod, slowperiod, matype)
_extra_function_map.update({f'ta_PPO_12_26': _Function(function=wrap_non_picklable_objects(lambda x: ta_PPO(x, 12, 26, 0)), name = 'ta_PPO_12_26', arity=1)})

@pre_and_post_process
def ta_RSI(x, timeperiod):
    return np.apply_along_axis(talib.RSI, 0, x, timeperiod)
_extra_function_map.update({f'ta_RSI_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_RSI(x, w)), name=f'ta_RSI_{w}', arity=1) for w in [10]})

def _stochrsi(x, timeperiod, fastk_period, fastd_period, fastd_matype, type):
    fastk, fastd = talib.STOCHRSI(x, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
    if type == 'k':
        return fastk
    elif type == 'd':
        return fastd
@pre_and_post_process
def ta_STOCHRSIk(x, timeperiod):
    return np.apply_along_axis(_stochrsi, 0, x, timeperiod=timeperiod, fastk_period=5, fastd_period=3, fastd_matype=0, type='k')
@pre_and_post_process
def ta_STOCHRSId(x, timeperiod):
    return np.apply_along_axis(_stochrsi, 0, x, timeperiod=timeperiod, fastk_period=5, fastd_period=3, fastd_matype=0, type='d')
_extra_function_map.update({f'ta_STOCHRSIk_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_STOCHRSIk(x, w)), name=f'ta_STOCHRSIk_{w}', arity=1) for w in [10]})
_extra_function_map.update({f'ta_STOCHRSId_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_STOCHRSId(x, w)), name=f'ta_STOCHRSId_{w}', arity=1) for w in [10]})

@pre_and_post_process
def ta_TRIX(x, timeperiod):
    return np.apply_along_axis(talib.TRIX, 0, x, timeperiod)
_extra_function_map.update({f'ta_TRIX_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_TRIX(x, w)), name=f'ta_TRIX_{w}', arity=1) for w in [20]})

# endregion

# region ==== TA functions: Overlap ====

def _bbands(x, timeperiod, nbdevup, nbdevdn, matype, type=''):
    u, m, l = talib.BBANDS(x, timeperiod, nbdevup, nbdevdn, matype)
    if type == 'u':
        return u
    elif type == 'm':
        return m
    elif type == 'l':
        return l
@pre_and_post_process
def ta_BBANDSu(x, timeperiod, nbdevup, nbdevdn, matype):
    return np.apply_along_axis(_bbands, 0, x, timeperiod, nbdevup, nbdevdn, matype, 'u')
@pre_and_post_process
def ta_BBANDSm(x, timeperiod, nbdevup, nbdevdn, matype):
    return np.apply_along_axis(_bbands, 0, x, timeperiod, nbdevup, nbdevdn, matype, 'm')
@pre_and_post_process
def ta_BBANDSl(x, timeperiod, nbdevup, nbdevdn, matype,):
    return np.apply_along_axis(_bbands, 0, x, timeperiod, nbdevup, nbdevdn, matype, 'l')
_extra_function_map.update({
    'ta_BBANDSu_5_2_2': _Function(function=wrap_non_picklable_objects(lambda x: ta_BBANDSu(x, 5, 2, 2, 0)), name = 'ta_BBANDSu_5_2_2', arity=1),
    'ta_BBANDSm_5_2_2': _Function(function=wrap_non_picklable_objects(lambda x: ta_BBANDSm(x, 5, 2, 2, 0)), name = 'ta_BBANDSm_5_2_2', arity=1),
    'ta_BBANDSl_5_2_2': _Function(function=wrap_non_picklable_objects(lambda x: ta_BBANDSl(x, 5, 2, 2, 0)), name = 'ta_BBANDSl_5_2_2', arity=1),
})

@pre_and_post_process
def ta_DEMA(x, timeperiod):
    return np.apply_along_axis(talib.DEMA, 0, x, timeperiod)
_extra_function_map.update({f'ta_DEMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_DEMA(x, w)), name=f'ta_DEMA_{w}', arity=1) for w in ts_wins if w >=5})

@pre_and_post_process
def ta_EMA(x, timeperiod):
    return np.apply_along_axis(talib.EMA, 0, x, timeperiod)
_extra_function_map.update({f'ta_EMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_EMA(x, w)), name=f'ta_EMA_{w}', arity=1) for w in ts_wins if w >=5})

@pre_and_post_process
def ta_HTTRENDLINE(x):
    return np.apply_along_axis(talib.HT_TRENDLINE, 0, x)
_extra_function_map.update({f'ta_HTTRENDLINE': _Function(function=wrap_non_picklable_objects(ta_HTTRENDLINE), name=f'ta_HTTRENDLINE', arity=1)})

def _mama(x, fastlimit, slowlimit, type=''):
    mama, fama = talib.MAMA(x, fastlimit, slowlimit)
    if type == 'mama':
        return mama
    elif type == 'fama':
        return fama
@pre_and_post_process
def ta_MAMA(x, fastlimit, slowlimit):
    return np.apply_along_axis(_mama, 0, x, fastlimit, slowlimit, 'mama')
@pre_and_post_process
def ta_FAMA(x, fastlimit, slowlimit):
    return np.apply_along_axis(_mama, 0, x, fastlimit, slowlimit, 'fama')
_extra_function_map.update({
    'ta_MAMA_0.5_0.05': _Function(function=wrap_non_picklable_objects(lambda x: ta_MAMA(x, 0.5, 0.05)), name = 'ta_MAMA_0.5_0.05', arity=1),
    'ta_FAMA_0.5_0.05': _Function(function=wrap_non_picklable_objects(lambda x: ta_FAMA(x, 0.5, 0.05)), name = 'ta_FAMA_0.5_0.05', arity=1),
})

@pre_and_post_process
def ta_SMA(x, timeperiod):
    return np.apply_along_axis(talib.SMA, 0, x, timeperiod)
_extra_function_map.update({f'ta_SMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_SMA(x, w)), name=f'ta_SMA_{w}', arity=1) for w in ts_wins if w >=5})

@pre_and_post_process
def ta_T3(x, timeperiod, vfactor):
    return np.apply_along_axis(talib.T3, 0, x, timeperiod, vfactor)
_extra_function_map.update({f'ta_T3_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_T3(x, w, 0.7)), name=f'ta_T3_{w}', arity=1) for w in ts_wins if w >=5})

@pre_and_post_process
def ta_TEMA(x, timeperiod):
    return np.apply_along_axis(talib.TEMA, 0, x, timeperiod)
_extra_function_map.update({f'ta_TEMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_TEMA(x, w)), name=f'ta_TEMA_{w}', arity=1) for w in ts_wins if w >=5})

@pre_and_post_process
def ta_TRIMA(x, timeperiod):
    return np.apply_along_axis(talib.TRIMA, 0, x, timeperiod)
_extra_function_map.update({f'ta_TRIMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_TRIMA(x, w)), name=f'ta_TRIMA_{w}', arity=1) for w in ts_wins if w >=5})

@pre_and_post_process
def ta_WMA(x, timeperiod):
    return np.apply_along_axis(talib.WMA, 0, x, timeperiod)
_extra_function_map.update({f'ta_WMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_WMA(x, w)), name=f'ta_WMA_{w}', arity=1) for w in ts_wins if w >=5})

# endregion

# region ==== TA functions: Volume ====

def _vwma(x1, x2, w):
    pv = x1 * x2
    vwma = talib.MA(pv, timeperiod=w, matype=0) / talib.MA(x2, timeperiod=w, matype=0)
    return vwma
@pre_and_post_process_2
def ta_VWMA(x1, x2, w=5):     ## use VWMA as an arity 2 func
    return apply_column_2(x1, x2, _vwma, w=w)
_extra_function_map.update({f'ta_VWMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_VWMA(x, w)), name=f'ta_VWMA_{w}', arity=2) for w in ts_wins if w >=5})

# endregion

# region ==== TA functions: Cycle ====

@pre_and_post_process
def ta_HTDCPERIOD(x):   # Hilbert Transform
    return np.apply_along_axis(talib.HT_DCPERIOD, 0, x)
_extra_function_map.update({f'ta_HTDCPERIOD': _Function(function=wrap_non_picklable_objects(ta_HTDCPERIOD), name=f'ta_HTDCPERIOD', arity=1)})

@pre_and_post_process
def ta_HTDCPHASE(x):   # Hilbert Transform
    return np.apply_along_axis(talib.HT_DCPHASE, 0, x)
_extra_function_map.update({f'ta_HTDCPHASE': _Function(function=wrap_non_picklable_objects(ta_HTDCPHASE), name=f'ta_HTDCPHASE', arity=1)})

def _ta_HTPHASOR(x, mode):
    inphase, quadrature = talib.HT_PHASOR(x)
    if mode == 'i':
        return inphase
    elif mode == 'q':
        return quadrature
@pre_and_post_process
def ta_HTPHASOR_INPHASE(x):   # Hilbert Transform
    return np.apply_along_axis(_ta_HTPHASOR, 0, x, mode='i')
_extra_function_map.update({f'ta_HTPHASOR_INPHASE': _Function(function=wrap_non_picklable_objects(ta_HTPHASOR_INPHASE), name=f'ta_HTPHASOR_INPHASE', arity=1)})

def _ta_HTSINE(x, mode):
    sine, leadsine = talib.HT_SINE(x)
    if mode == 's':
        return sine
    elif mode == 'l':
        return leadsine
@pre_and_post_process
def ta_HTSINE_SINE(x):   # Hilbert Transform
    return np.apply_along_axis(_ta_HTSINE, 0, x, mode='s')
_extra_function_map.update({f'ta_HTSINE_SINE': _Function(function=wrap_non_picklable_objects(ta_HTSINE_SINE), name=f'ta_HTSINE_SINE', arity=1)})
@pre_and_post_process
def ta_HTSINE_LEADSINE(x):   # Hilbert Transform
    return np.apply_along_axis(_ta_HTSINE, 0, x, mode='l')
_extra_function_map.update({f'ta_HTSINE_LEADSINE': _Function(function=wrap_non_picklable_objects(ta_HTSINE_LEADSINE), name=f'ta_HTSINE_LEADSINE', arity=1)})

@pre_and_post_process
def ta_HTTRENDMODE(x):   # Hilbert Transform
    return np.apply_along_axis(talib.HT_TRENDMODE, 0, x).astype(np.float64)
_extra_function_map.update({f'ta_HTTRENDMODE': _Function(function=wrap_non_picklable_objects(ta_HTTRENDMODE), name=f'ta_HTTRENDMODE', arity=1)})


# endregion

# region ==== TA functions: Statistic ====

@pre_and_post_process_2
def ta_BETA(x1, x2, w):
    return apply_column_2(x1, x2, talib.BETA, timeperiod=w)
_extra_function_map.update({f'ta_BETA_{w}': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: ta_BETA(x1, x2, w)), name=f'ta_BETA_{w}', arity=2) for w in ts_wins if w >=5})

@pre_and_post_process_2
def ta_CORREL(x1, x2, w):   ## Pearson's Correlation Coefficient (r)
    return apply_column_2(x1, x2, talib.CORREL, timeperiod=w)
_extra_function_map.update({f'ta_CORREL_{w}': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: ta_CORREL(x1, x2, w)), name=f'ta_CORREL_{w}', arity=2) for w in ts_wins if w >=10})

@pre_and_post_process
def ta_LINEARREG(x, w=10):  ## Linear Regression
    return np.apply_along_axis(talib.LINEARREG, 0, x, timeperiod=w)
_extra_function_map.update({f'ta_LINEARREG_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_LINEARREG(x, w)), name=f'ta_LINEARREG_{w}', arity=1) for w in ts_wins if w >=10})

@pre_and_post_process
def ta_LINEARREG_ANGLE(x, w=10):  ## Linear Regression Angle
    return np.apply_along_axis(talib.LINEARREG_ANGLE, 0, x, timeperiod=w)
_extra_function_map.update({f'ta_LINEARREG_ANGLE_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_LINEARREG_ANGLE(x, w)), name=f'ta_LINEARREG_ANGLE_{w}', arity=1) for w in ts_wins if w >=10})

@pre_and_post_process
def ta_LINEARREG_INTERCEPT(x, w=10):  ## Linear Regression Intercept
    return np.apply_along_axis(talib.LINEARREG_INTERCEPT, 0, x, timeperiod=w)
_extra_function_map.update({f'ta_LINEARREG_INTERCEPT_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_LINEARREG_INTERCEPT(x, w)), name=f'ta_LINEARREG_INTERCEPT_{w}', arity=1) for w in ts_wins if w >=10})

@pre_and_post_process
def ta_LINEARREG_SLOPE(x, w=10):  ## Linear Regression Slope
    return np.apply_along_axis(talib.LINEARREG_SLOPE, 0, x, timeperiod=w)
_extra_function_map.update({f'ta_LINEARREG_SLOPE_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_LINEARREG_SLOPE(x, w)), name=f'ta_LINEARREG_SLOPE_{w}', arity=1) for w in ts_wins if w >=10})

@pre_and_post_process
def ta_TSF(x, w=10):  ## Time Series Forecast
    return np.apply_along_axis(talib.TSF, axis=0, arr=x, timeperiod=w)
_extra_function_map.update({f'ta_TSF_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_TSF(x, w)), name=f'ta_TSF_{w}', arity=1) for w in ts_wins if w >=10})

## Mean absolute deviation around the median (not mean)
@pre_and_post_process
def ta_MAD(x, w=5):
    rolling_median = bn.move_mean(x, window=w, min_count=int(w/2), axis=0)
    abs_dev = np.fabs(x - rolling_median)
    mad = bn.move_mean(abs_dev, window=w, min_count=int(w/2), axis=0)
    return mad
_extra_function_map.update({f'ta_MAD_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_MAD(x, w)), name=f'ta_MAD_{w}', arity=1) for w in ts_wins if w >=5})


# endregion

# region ==== TA functions: Math Transform ====

@pre_and_post_process
def ta_ATAN(x):  
    return np.apply_along_axis(talib.ATAN, 0, x)
_extra_function_map.update({f'ta_ATAN': _Function(function=wrap_non_picklable_objects(ta_ATAN), name=f'ta_ATAN', arity=1)})

@pre_and_post_process
def ta_CEIL(x):  
    return np.apply_along_axis(talib.CEIL, 0, x)
_extra_function_map.update({f'ta_CEIL': _Function(function=wrap_non_picklable_objects(ta_CEIL), name=f'ta_CEIL', arity=1)})

@pre_and_post_process
def ta_COS(x):  
    return np.apply_along_axis(talib.COS, 0, x)
_extra_function_map.update({f'ta_COS': _Function(function=wrap_non_picklable_objects(ta_COS), name=f'ta_COS', arity=1)})

@pre_and_post_process
def ta_COSH(x):  
    return np.apply_along_axis(talib.COSH, 0, x)
_extra_function_map.update({f'ta_COSH': _Function(function=wrap_non_picklable_objects(ta_COSH), name=f'ta_COSH', arity=1)})

@pre_and_post_process
def ta_FLOOR(x):  
    return np.apply_along_axis(talib.FLOOR, 0, x)
_extra_function_map.update({f'ta_FLOOR': _Function(function=wrap_non_picklable_objects(ta_FLOOR), name=f'ta_FLOOR', arity=1)})

@pre_and_post_process
def ta_SIN(x):  
    return np.apply_along_axis(talib.SIN, 0, x)
_extra_function_map.update({f'ta_SIN': _Function(function=wrap_non_picklable_objects(ta_SIN), name=f'ta_SIN', arity=1)})

@pre_and_post_process
def ta_SINH(x):  
    return np.apply_along_axis(talib.SINH, 0, x)
_extra_function_map.update({f'ta_SINH': _Function(function=wrap_non_picklable_objects(ta_SINH), name=f'ta_SINH', arity=1)})

@pre_and_post_process
def ta_TAN(x):  
    return np.apply_along_axis(talib.TAN, 0, x)
_extra_function_map.update({f'ta_TAN': _Function(function=wrap_non_picklable_objects(ta_TAN), name=f'ta_TAN', arity=1)})

@pre_and_post_process
def ta_TANH(x):  
    return np.apply_along_axis(talib.TANH, 0, x)
_extra_function_map.update({f'ta_TANH': _Function(function=wrap_non_picklable_objects(ta_TANH), name=f'ta_TANH', arity=1)})

# endregion

# endregion TA


# region ==== Basic ====

# region ==== Basic: arity > 1, w/ & w/o ts zscore ====

@pre_and_post_process_2
def add(x1, x2):
    return np.add(x1, x2)
_extra_function_map.update({f'add': _Function(function=wrap_non_picklable_objects(add), name=f'add', arity=2)})

@pre_and_post_process_2
def cszs_add(x1, x2):
    return np.add(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_add': _Function(function=wrap_non_picklable_objects(cszs_add), name=f'cszs_add', arity=2)})

@pre_and_post_process_2
def tszs_add(x1, x2, w):
    return np.add(x1, x2)
_extra_function_map.update({f'tszs_{w}_add': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_add(x1, x2, w)), name=f'tszs_{w}_add', arity=2) for w in tszs_wins})


@pre_and_post_process_2
def sub(x1, x2):
    return np.subtract(x1, x2)
_extra_function_map.update({f'sub': _Function(function=wrap_non_picklable_objects(sub), name=f'sub', arity=2)})

@pre_and_post_process_2
def cszs_sub(x1, x2):
    return np.subtract(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_sub': _Function(function=wrap_non_picklable_objects(cszs_sub), name=f'cszs_sub', arity=2)})

@pre_and_post_process_2
def tszs_sub(x1, x2, w):
    return np.subtract(x1, x2)
_extra_function_map.update({f'tszs_{w}_sub': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_sub(x1, x2, w)), name=f'tszs_{w}_sub', arity=2) for w in tszs_wins})


@pre_and_post_process_2
def mul(x1, x2):
    return np.multiply(x1, x2)
_extra_function_map.update({f'mul': _Function(function=wrap_non_picklable_objects(mul), name=f'mul', arity=2)})

@pre_and_post_process_2
def cszs_mul(x1, x2):
    return np.multiply(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_mul': _Function(function=wrap_non_picklable_objects(cszs_mul), name=f'cszs_mul', arity=2)})

@pre_and_post_process_2
def tszs_mul(x1, x2, w):
    return np.multiply(x1, x2)
_extra_function_map.update({f'tszs_{w}_mul': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_mul(x1, x2, w)), name=f'tszs_{w}_mul', arity=2) for w in tszs_wins})


@pre_and_post_process_2
def div(x1, x2):
    return np.where(x2 > 0.001, np.divide(x1, x2), 1.)
_extra_function_map.update({f'div': _Function(function=wrap_non_picklable_objects(div), name=f'div', arity=2)})

@pre_and_post_process_2
def cszs_div(x1, x2):
    cszs_x2 = cs_zscore(x2)
    return np.where(cszs_x2 > 0.001, np.divide(cs_zscore(x1), cszs_x2), 1.)
_extra_function_map.update({f'cszs_div': _Function(function=wrap_non_picklable_objects(cszs_div), name=f'cszs_div', arity=2)})

@pre_and_post_process_2
def tszs_div(x1, x2, w):
    return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)
_extra_function_map.update({f'tszs_{w}_div': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_div(x1, x2, w)), name=f'tszs_{w}_div', arity=2) for w in tszs_wins})

@pre_and_post_process_2
def max(x1, x2):
    return np.maximum(x1, x2)
_extra_function_map.update({f'max': _Function(function=wrap_non_picklable_objects(max), name=f'max', arity=2)})

@pre_and_post_process_2
def cszs_max(x1, x2):
    return np.maximum(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_max': _Function(function=wrap_non_picklable_objects(cszs_max), name=f'cszs_max', arity=2)})

@pre_and_post_process_2
def tszs_max(x1, x2, w):
    return np.maximum(x1, x2)
_extra_function_map.update({f'tszs_{w}_max': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_max(x1, x2, w)), name=f'tszs_{w}_max', arity=2) for w in tszs_wins})


@pre_and_post_process_2
def min(x1, x2):
    return np.minimum(x1, x2)
_extra_function_map.update({f'min': _Function(function=wrap_non_picklable_objects(min), name=f'min', arity=2)})

@pre_and_post_process_2
def cszs_min(x1, x2):
    return np.minimum(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_min': _Function(function=wrap_non_picklable_objects(cszs_min), name=f'cszs_min', arity=2)})

@pre_and_post_process_2
def tszs_min(x1, x2, w):
    return np.minimum(x1, x2)
_extra_function_map.update({f'tszs_{w}_min': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_min(x1, x2, w)), name=f'tszs_{w}_min', arity=2) for w in tszs_wins})

# endregion

# region ==== Basic: arity = 1, simple ====

@pre_and_post_process
def sqrt(x):
    return np.sqrt(np.abs(x))
_extra_function_map.update({f'sqrt': _Function(function=wrap_non_picklable_objects(sqrt), name=f'sqrt', arity=1)})

@pre_and_post_process
def signed_sqrt(x):
    return np.sqrt(np.abs(x)) * np.sign(x)
_extra_function_map.update({f'signed_sqrt': _Function(function=wrap_non_picklable_objects(signed_sqrt), name=f'signed_sqrt', arity=1)})

@pre_and_post_process
def log(x):
    return np.where(np.abs(x) > 0.0001, np.log(np.abs(x)), 0.)
_extra_function_map.update({f'log': _Function(function=wrap_non_picklable_objects(log), name=f'log', arity=1)})

@pre_and_post_process
def signed_log(x):
    return np.log(np.abs(x)) * np.sign(x)
_extra_function_map.update({f'signed_log': _Function(function=wrap_non_picklable_objects(signed_log), name=f'signed_log', arity=1)})

@pre_and_post_process
def neg(x):
    return np.negative(x)
_extra_function_map.update({f'neg': _Function(function=wrap_non_picklable_objects(neg), name=f'neg', arity=1)})

@pre_and_post_process
def inv(x):
    return np.where(np.abs(x) > 0.0001, 1. / x, 0.)
_extra_function_map.update({f'inv': _Function(function=wrap_non_picklable_objects(inv), name=f'inv', arity=1)})

@pre_and_post_process
def abs(x):
    return np.abs(x)
_extra_function_map.update({f'abs': _Function(function=wrap_non_picklable_objects(abs), name=f'abs', arity=1)})

@pre_and_post_process
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
_extra_function_map.update({f'sigmoid': _Function(function=wrap_non_picklable_objects(sigmoid), name=f'sigmoid', arity=1)})

@pre_and_post_process
def pow2(x):
    return np.power(x, 2)
_extra_function_map.update({f'pow2': _Function(function=wrap_non_picklable_objects(pow2), name=f'pow2', arity=1)})

# endregion

# endregion Basic


# region ==== Multi-conditions ====

@pre_and_post_process_3
def clear_by_cond(x1, x2, x3):
    """if x1 < x2 (keep NaN if and only if both x1 and x2 are NaN), then 0, else x3"""
    return np.where(x1 < x2, 0, np.where(~np.isnan(x1) | ~np.isnan(x2), x3, np.nan))
_extra_function_map.update({'clear_by_cond': _Function(function=wrap_non_picklable_objects(clear_by_cond), name="clear_by_cond", arity=3)})

pre_and_post_process_3
def if_then_else(x1, x2, x3):
    """if x1 is nonzero (keep NaN), then x2, else x3"""
    return np.where(x1, x2, np.where(~np.isnan(x1), x3, np.nan))
_extra_function_map.update({'if_then_else': _Function(function=wrap_non_picklable_objects(if_then_else), name="if_then_else", arity=3)})

@pre_and_post_process_4
def if_cond_then_else(x1, x2, x3, x4):
    """if x1 < x2 (keep NaN if and only if both x1 and x2 are NaN), then x3, else x4"""
    return np.where(x1 < x2, x3, np.where(~np.isnan(x1) | ~np.isnan(x2), x4, np.nan))
_extra_function_map.update({'if_cond_then_else': _Function(function=wrap_non_picklable_objects(if_cond_then_else), name="if_cond_then_else", arity=4)})

@pre_and_post_process_3
def if_above0_then_else(x1, x2, x3):
    return np.where(np.isnan(x1), x3, np.where(x1 > 0, x2, x3))
_extra_function_map.update({'if_above0_then_else': _Function(function=wrap_non_picklable_objects(if_above0_then_else), name="if_above0_then_else", arity=3)})

@pre_and_post_process_2
def if_nan_then_else(x1, x2):
    return np.where(np.isnan(x1), x2, x1)
_extra_function_map.update({'if_nan_then_else': _Function(function=wrap_non_picklable_objects(if_nan_then_else), name="if_nan_then_else", arity=2)})

@pre_and_post_process_2
def if_larger_then_else(x1, x2):
    return np.where(x1 > x2, x1, x2)
_extra_function_map.update({'if_larger_then_else': _Function(function=wrap_non_picklable_objects(if_larger_then_else), name="if_larger_then_else", arity=2)})

# endregion


# region ======= valuable function but use them under conditions

## [Test simulate live delta update] failed (on 500 lookback)
## [Test simulate live delta update] passed (on 2000 lookback) -- use it carefully
@pre_and_post_process
def ta_KAMA(x, timeperiod):
    # x = nbgg.ffill(x)
    # x = np.where(np.isnan(x), 0, x)
    return np.apply_along_axis(talib.KAMA, 0, x, timeperiod)
_extra_function_map.update({f'ta_KAMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_KAMA(x, w)), name=f'ta_KAMA_{w}', arity=1) for w in ts_wins if w >=5})

# endregion


# region ======= disabled functions

## [Test startpoint dependency] failed
from pandas_ta.cycles.reflex import np_reflex
@pre_and_post_process
def ta_REFLEX(x, w=10):
    ret = np.apply_along_axis(np_reflex, 0, x, n=w, k=w, alpha=0.04, pi=3.14159, sqrt2=1.414)
    ret[:w, :] = np.nan
    return ret
# _extra_function_map.update({f'ta_REFLEX_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_REFLEX(x, w)), name=f'ta_REFLEX_{w}', arity=1) for w in ts_wins if w >=10})

## [Test high nan ratio] failed (1.0 on 0.2 nan rate)
def ta_HTPHASOR_QUADRATURE(x):   # Hilbert Transform
    return np.apply_along_axis(_ta_HTPHASOR, 0, x, mode='q')
# _extra_function_map.update({f'ta_HTPHASOR_QUADRATURE': _Function(function=wrap_non_picklable_objects(ta_HTPHASOR_QUADRATURE), name=f'ta_HTPHASOR_QUADRATURE', arity=1)})

## [Test high nan ratio] failed (0.6 on 0.2 nan rate)
@pre_and_post_process
def ta_LN(x):  
    return np.apply_along_axis(talib.LN, 0, x)
# _extra_function_map.update({f'ta_LN': _Function(function=wrap_non_picklable_objects(ta_LN), name=f'ta_LN', arity=1)})

## [Test high nan ratio] failed (0.4 on 0.2 nan rate5)
@pre_and_post_process
def ta_ACOS(x):  
    return np.apply_along_axis(talib.ACOS, 0, x)
# _extra_function_map.update({f'ta_ACOS': _Function(function=wrap_non_picklable_objects(ta_ACOS), name=f'ta_ACOS', arity=1)})

## [Test high nan ratio] failed (0.45 on 0.2 nan rate)
@pre_and_post_process
def ta_ASIN(x):  
    return np.apply_along_axis(talib.ASIN, 0, x)
# _extra_function_map.update({f'ta_ASIN': _Function(function=wrap_non_picklable_objects(ta_ASIN), name=f'ta_ASIN', arity=1)})


## [Test startpoint dependency] failed
@pre_and_post_process_2
def ta_OBV(x1, x2):     ## use OBV as an arity 2 func
    return apply_column_2(x1, x2, talib.OBV)
# _extra_function_map.update({f'ta_OBV': _Function(function=wrap_non_picklable_objects(ta_OBV), name=f'ta_OBV', arity=2)})

@pre_and_post_process_2
def cszs_ta_OBV(x1, x2):     ## use OBV as an arity 2 func
    return apply_column_2(cs_zscore(x1), cs_zscore(x2), talib.OBV)
# _extra_function_map.update({f'cszs_ta_OBV': _Function(function=wrap_non_picklable_objects(cszs_ta_OBV), name=f'cszs_ta_OBV', arity=2)})

@pre_and_post_process_2
def tszs_ta_OBV(x1, x2, w):     ## use OBV as an arity 2 func
    return apply_column_2(x1, x2, talib.OBV)
# _extra_function_map.update({f'tszs_{w}_ta_OBV': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_ta_OBV(x1, x2, w)), name=f'tszs_{w}_ta_OBV', arity=2) for w in tszs_wins})


# endregion