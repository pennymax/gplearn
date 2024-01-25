import numpy as np
import pandas as pd

from .functions import _Function, _protected_division
from joblib import wrap_non_picklable_objects
import pandas_ta as ta
import talib



def error_state_decorator(func):
    def wrapper(A, *args, **kwargs):
        with np.errstate(over='ignore', under='ignore'):
            return func(A, *args, **kwargs)
    return wrapper

def apply_column(x, func, **kwargs):
    r = np.empty_like(x)
    for i in range(x.shape[1]):
        r[:, i] = func(x[:, i], **kwargs)
    return r

@error_state_decorator
def ta_APO(x, w): 
    return apply_column(x, talib.APO, timeperiod=w)

@error_state_decorator
def ta_BOP(x, w):
    return apply_column(x, talib.BOP, timeperiod=w)

error_state_decorator
# def ta_cci(h, l, c, l)    ## tbd


def _macd(x, fast, slow, signal):
    m, s, h = talib.MACD(x, fast=fast, slow=slow, signal=signal)
    return m
def _macds(x, fast, slow, signal):
    m, s, h = talib.MACD(x, fast=fast, slow=slow, signal=signal)
    return s
def _macdh(x, fast, slow, signal):
    m, s, h = talib.MACD(x, fast=fast, slow=slow, signal=signal)
    return h

@error_state_decorator
def ta_MACD(x, fast, slow, signal):
    return apply_column(x, _macd, fast=fast, slow=slow, signal=signal)
def ta_MACDs(x, fast, slow, signal):
    return apply_column(x, _macds, fast=fast, slow=slow, signal=signal)
def ta_MACDh(x, fast, slow, signal):
    return apply_column(x, _macdh, fast=fast, slow=slow, signal=signal)


def ts_zscore(x, w):
    # 初始化结果数组
    r = np.empty_like(x)
    r[:] = np.nan

    # 计算 rolling z-score
    for i in range(w - 1, x.shape[0]):
        window_slice = x[i - w + 1:i + 1, :]
        mean = np.nanmean(window_slice, axis=0)
        std = np.nanstd(window_slice, axis=0)
        r[i, :] = (x[i, :] - mean) / std
    return r

@error_state_decorator
def tszs_add(x1, x2, w):
    tszs_x1 = apply_column(x1, ts_zscore, w=w)
    tszs_x2 = apply_column(x2, ts_zscore, w=w)
    return np.add(tszs_x1, tszs_x2)