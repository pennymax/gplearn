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


@nb.jit(nopython=True)
def compute_autocorr(window, lag, window_size):
    # 检查窗口长度是否足够
    if len(window) < window_size + lag:
        return np.nan

    # 使用完整的window_size进行计算
    y = window[lag:lag+window_size]
    y_lag = window[:window_size]

    # 计算均值
    mean_y = np.nanmean(y)
    mean_y_lag = np.nanmean(y_lag)

    # 计算协方差和方差
    cov = np.nansum((y - mean_y) * (y_lag - mean_y_lag))
    var_y = np.nansum((y - mean_y) ** 2)
    var_y_lag = np.nansum((y_lag - mean_y_lag) ** 2)

    if var_y == 0 or var_y_lag == 0:
        return np.nan  # 避免除以零

    correlation = cov / np.sqrt(var_y * var_y_lag)
    return correlation

@nb.jit(nopython=True)
def rolling_autocorr(data, window_size, lag):
    n = data.shape[0]
    cols = data.shape[1]
    result = np.full((n, cols), np.nan)  # 初始化结果数组为 NaN

    # 从 index = window_size - 1 + lag 开始计算，以确保足够的数据
    for j in range(cols):
        for i in range(window_size - 1 + lag, n):
            window = data[i-window_size-lag+1:i+1, j]
            result[i, j] = compute_autocorr(window, lag, window_size)
    return result

@error_handle_and_nan_mask
def ts_autocorr(x, w=5, l=1):   # no faster way for now
    return rolling_autocorr(x, w, l)



@nb.jit(nopython=True, cache=True)
def rolling_apply_mad(data, window_size):
    n = data.shape[0]
    cols = data.shape[1]
    result = np.empty((n, cols))
    result[:] = np.nan

    for j in range(cols):
        for i in range(window_size - 1, n):
            window = data[i-window_size+1:i+1, j]
            valid_data = window[~np.isnan(window)]
            if len(valid_data) > 0:
                mean = valid_data.mean()
                mad = np.mean(np.abs(valid_data - mean))
                result[i, j] = mad

    return result

@nb.jit(nopython=True, cache=True)
def rolling_apply_mad_2(data, window_size):
    n = data.shape[0]
    cols = data.shape[1]
    result = np.empty((n, cols))
    result[:] = np.nan
    # 处理每一列
    for j in range(cols):
        # 计算每个窗口的 MAD
        for i in range(window_size - 1, n):
            mean = np.nanmean(data[i-window_size+1:i+1, j])
            result[i, j] = np.nanmean(np.abs(data[i-window_size+1:i+1, j] - mean))
    return result

## Mean absolute deviation around the median (not mean)
@error_handle_and_nan_mask
def ta_MAD(x, w=5):
    return rolling_apply_mad(x, w)



@error_handle_and_nan_mask
def ts_quantile(x, w=5, q=0.25):
    slide_view = sliding_window_view(x, w, axis=0)
    ret = np.full_like(x, np.nan)
    # ret[w-1:, :] = np.nanpercentile(slide_view, q*100, axis=2)
    ret[w-1:, :] = nbgg.nanquantile(slide_view, q, axis=2)
    return ret