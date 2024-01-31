import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import functools

from .functions import _Function
from joblib import wrap_non_picklable_objects
import pandas_ta as ta
import talib
from talib import MA_Type ## MA_Type.SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3



_extra_function_map= {}
tszs_wins = [60, 120]
ts_wins = [1, 3, 5, 10, 20, 40, 60]
ts_lags = [1, 3, 5, 10] # used in like autocorr, etc.



# region ==== Utils ====

def error_handle_and_nan_mask(func):
    @functools.wraps(func)
    def wrapper(A, *args, **kwargs):
        # 保存原始的 NaN 位置
        nan_mask = np.isnan(A)

        # 处理 NumPy 的错误状态
        with np.errstate(over='ignore', under='ignore'):
            result = func(A, *args, **kwargs)

        # 应用 NaN 掩码
        assert(result.shape == A.shape)
        result[nan_mask] = np.nan

        return result
    return wrapper

def apply_column(x, func, *args, **kwargs):
    r = np.empty_like(x)
    for i in range(x.shape[1]):
        r[:, i] = func(x[:, i], *args, **kwargs)
    return r

def apply_column_2(x1, x2, func, *args, **kwargs):
    r = np.empty_like(x1)
    for i in range(x1.shape[1]):
        r[:, i] = func(x1[:, i], x2[:, i], *args, **kwargs)
    return r

## quite faster than rolling.apply() but slower than rolling.pd_support_func()
def np_rolling_apply(x, w, func):
    if w > x.shape[0]:
        return np.full(x.shape, np.nan)

    roll_view = sliding_window_view(x, window_shape=w, axis=0)  # axis=0 split windows vertically (ts) 

    result = np.empty((x.shape[0], x.shape[1]))
    result[:w - 1] = np.nan  # 前 window - 1 个值设为 NaN

    for i in range(w - 1, x.shape[0]):
        result[i] = func(roll_view[i - w + 1], axis=1)  # apply func on each window horizontally. note: each window is a row in the roll_view!

    return result

# endregion


# region ==== Time Series functions ====

@error_handle_and_nan_mask
def ts_zscore(x, w=60):
    df = pd.DataFrame(x)
    rolling = df.rolling(window=w)
    mean = rolling.mean()
    std = rolling.std()

    # 避免除以零：如果 std 是 0，设置结果为 NaN
    zscore = (df - mean) / std
    zscore[std == 0] = np.nan

    return zscore.to_numpy(np.double)
_extra_function_map.update({f'tszs_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_zscore(x, w)), name=f'tszs_{w}', arity=1) for w in tszs_wins})

@error_handle_and_nan_mask
def ts_delay(x, w=1):
    return pd.DataFrame(x).shift(w).to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_delay_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_delay(x, w)), name=f'ts_delay_{w}', arity=1) for w in ts_wins})

@error_handle_and_nan_mask
def ts_diff(x, w=1):
    return pd.DataFrame(x).diff(w).to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_diff_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_diff(x, w)), name=f'ts_diff_{w}', arity=1) for w in ts_wins})

@error_handle_and_nan_mask
def ts_roc(x, w=1): # rate of change
    ret = pd.DataFrame(x)
    shifted_ret = ret.shift(w)
    return np.where(shifted_ret == 0, np.nan, ret.diff(w) / shifted_ret.abs() - 1)  # do not use pct_change as it can not handle negative values as expect
_extra_function_map.update({f'ts_roc_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_roc(x, w)), name=f'ts_roc_{w}', arity=1) for w in ts_wins})

@error_handle_and_nan_mask
def ts_sum(x, w=3):
    return pd.DataFrame(x).rolling(w).sum().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_sum_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_sum(x, w)), name=f'ts_sum_{w}', arity=1) for w in ts_wins if w > 1})

@error_handle_and_nan_mask
def ts_mean(x, w=3):
    return pd.DataFrame(x).rolling(w).mean().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_mean_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_mean(x, w)), name=f'ts_mean_{w}', arity=1) for w in ts_wins if w > 1})

@error_handle_and_nan_mask
def ts_median(x, w=3):
    return pd.DataFrame(x).rolling(w).median().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_median_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_median(x, w)), name=f'ts_median_{w}', arity=1) for w in ts_wins if w > 1})

@error_handle_and_nan_mask
def ts_std(x, w=5):
    return pd.DataFrame(x).rolling(w).std().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_std_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_std(x, w)), name=f'ts_std_{w}', arity=1) for w in ts_wins if w >= 5})

@error_handle_and_nan_mask
def ts_skew(x, w=5):
    return pd.DataFrame(x).rolling(w).skew().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_skew_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_skew(x, w)), name=f'ts_skew_{w}', arity=1) for w in ts_wins if w >= 5})

@error_handle_and_nan_mask
def ts_kurt(x, w=5):
    return pd.DataFrame(x).rolling(w).kurt().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_kurt_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_kurt(x, w)), name=f'ts_kurt_{w}', arity=1) for w in ts_wins if w >= 5})

@error_handle_and_nan_mask
def ts_max(x, w=5):
    return pd.DataFrame(x).rolling(w).max().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_max_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_max(x, w)), name=f'ts_max_{w}', arity=1) for w in ts_wins if w > 1})

@error_handle_and_nan_mask
def ts_min(x, w=5):
    return pd.DataFrame(x).rolling(w).min().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_min_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_min(x, w)), name=f'ts_min_{w}', arity=1) for w in ts_wins if w > 1})

@error_handle_and_nan_mask
def ts_rank(x, w=3):
    return pd.DataFrame(x).rolling(w).rank(pct=True).to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_rank_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_rank(x, w)), name=f'ts_rank_{w}', arity=1) for w in ts_wins if w >= 3})

@error_handle_and_nan_mask
def ts_autocorr(x, w=5, l=1):
    """moving autocorrelation coefficient between x and x lag i period"""
    ret = pd.DataFrame(x)
    ret_lag = ret.shift(l)
    return ret.rolling(w).corr(ret_lag).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_autocorr_{w}_{l}': _Function(function=wrap_non_picklable_objects(lambda x, w=w, l=l: ts_autocorr(x, w, l)), name=f'ts_autocorr_{w}_{l}', arity=1) for w in ts_wins for l in ts_lags if w >=3 and l <= w})

def safe_nanargmin(x, axis):
    x = x.copy()
    all_nan_col = np.all(np.isnan(x), axis=axis)
    x[all_nan_col, :] = 0
    ret = np.nanargmin(x, axis=axis).astype(float)
    ret[all_nan_col] = np.nan
    return ret

@error_handle_and_nan_mask
def ts_argmin(x, w=3):
    return np_rolling_apply(x, w, safe_nanargmin)
_extra_function_map.update({f'ts_argmin_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_argmin(x, w)), name=f'ts_argmin_{w}', arity=1) for w in ts_wins if w >=3})

def safe_nanargmax(x, axis):
    x = x.copy()
    all_nan_col = np.all(np.isnan(x), axis=axis)
    x[all_nan_col, :] = 0
    ret = np.nanargmax(x, axis=axis).astype(float)
    ret[all_nan_col] = np.nan
    return ret

@error_handle_and_nan_mask
def ts_argmax(x, w=3):
    return np_rolling_apply(x, w, safe_nanargmax)
_extra_function_map.update({f'ts_argmax_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_argmax(x, w)), name=f'ts_argmax_{w}', arity=1) for w in ts_wins if w >=3})

@error_handle_and_nan_mask
def ts_quantile(x, w=5, q=0.25):
    return pd.DataFrame(x).rolling(w).quantile(q).to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_quantile_{w}_0.25': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_quantile(x, w, 0.25)), name=f'ts_quantile_{w}_0.25', arity=1) for w in ts_wins if w >=5})
_extra_function_map.update({f'ts_quantile_{w}_0.75': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_quantile(x, w, 0.75)), name=f'ts_quantile_{w}_0.75', arity=1) for w in ts_wins if w >=5})

@error_handle_and_nan_mask
def ts_decreasing(x, w=5):
    return pd.DataFrame(x).apply(lambda col: ta.decreasing(col, length=w, asint=True)).to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_decreasing_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_decreasing(x, w)), name=f'ts_decreasing_{w}', arity=1) for w in ts_wins if w >=3})

@error_handle_and_nan_mask
def ts_increasing(x, w=5):
    return pd.DataFrame(x).apply(lambda col: ta.increasing(col, length=w, asint=True)).to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_increasing_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_increasing(x, w)), name=f'ts_increasing_{w}', arity=1) for w in ts_wins if w >=3})

#endregion


# region ==== Cross Section functions ====

@error_handle_and_nan_mask
def cs_rank(x):
    return pd.DataFrame(x).rank(axis=1, pct=True).to_numpy(dtype=np.double)
_extra_function_map.update({'cs_rank': _Function(function=wrap_non_picklable_objects(lambda x: cs_rank(x)), name = 'cs_rank', arity=1)})

@error_handle_and_nan_mask
def cs_zscore(x):
    # 检查每行是否全部为 NaN
    all_nan_rows = np.isnan(x).all(axis=1)

    # 为非全 NaN 的行计算均值和标准差
    means = np.nanmean(x[~all_nan_rows], axis=1, keepdims=True)
    stds = np.nanstd(x[~all_nan_rows], axis=1, keepdims=True)

    # 初始化一个全为 NaN 的结果数组
    z_scores = np.full(x.shape, np.nan)

    # 只对非全 NaN 的行进行 Z-score 计算
    z_scores[~all_nan_rows] = (x[~all_nan_rows] - means) / stds

    return z_scores
_extra_function_map.update({'cs_zscore': _Function(function=wrap_non_picklable_objects(lambda x: cs_zscore(x)), name = 'cs_zscore', arity=1)})


# endregion


# region ==== TA functions ====

# region ==== TA functions: Momentum ====

@error_handle_and_nan_mask
def ta_APO(x, fastperiod, slowperiod, matype):
    return apply_column(x, talib.APO, fastperiod, slowperiod, matype)
_extra_function_map.update({f'ta_APO_12_26': _Function(function=wrap_non_picklable_objects(lambda x: ta_APO(x, 12, 26, 0)), name = 'ta_APO_12_26', arity=1)})

@error_handle_and_nan_mask
def ta_CMO(x, timeperiod):
    return apply_column(x, talib.CMO, timeperiod)
_extra_function_map.update({f'ta_CMO_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_CMO(x, w)), name=f'ta_CMO_{w}', arity=1) for w in [10]})

def _macd(x, fastperiod, slowperiod, signalperiod, type=''):
    m, s, h = talib.MACD(x, fastperiod, slowperiod, signalperiod)
    if type == 's':
        return s
    elif type == 'h':
        return h
    else:
        return m
@error_handle_and_nan_mask
def ta_MACD(x, fastperiod, slowperiod, signalperiod):
    return apply_column(x, _macd, fastperiod, slowperiod, signalperiod, '')
@error_handle_and_nan_mask
def ta_MACDs(x, fastperiod, slowperiod, signalperiod):
    return apply_column(x, _macd, fastperiod, slowperiod, signalperiod, 's')
@error_handle_and_nan_mask
def ta_MACDh(x, fastperiod, slowperiod, signalperiod):
    return apply_column(x, _macd, fastperiod, slowperiod, signalperiod, 'h')
_extra_function_map.update({
    'ta_MACD_12_26_9': _Function(function=wrap_non_picklable_objects(lambda x: ta_MACD(x, 12, 26, 9)), name = 'ta_MACD_12_26_9', arity=1),
    'ta_MACDs_12_26_9': _Function(function=wrap_non_picklable_objects(lambda x: ta_MACDs(x, 12, 26, 9)), name = 'ta_MACDs_12_26_9', arity=1),
    'ta_MACDh_12_26_9': _Function(function=wrap_non_picklable_objects(lambda x: ta_MACDh(x, 12, 26, 9)), name = 'ta_MACDh_12_26_9', arity=1),
})

@error_handle_and_nan_mask
def ta_PPO(x, fastperiod, slowperiod, matype):
    return apply_column(x, talib.PPO, fastperiod, slowperiod, matype)
_extra_function_map.update({f'ta_PPO_12_26': _Function(function=wrap_non_picklable_objects(lambda x: ta_PPO(x, 12, 26, 0)), name = 'ta_PPO_12_26', arity=1)})

@error_handle_and_nan_mask
def ta_RSI(x, timeperiod):
    return apply_column(x, talib.RSI, timeperiod)
_extra_function_map.update({f'ta_RSI_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_RSI(x, w)), name=f'ta_RSI_{w}', arity=1) for w in [10]})


def _stochrsi(x, timeperiod, fastk_period, fastd_period, fastd_matype, type):
    fastk, fastd = talib.STOCHRSI(x, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
    if type == 'k':
        return fastk
    elif type == 'd':
        return fastd
@error_handle_and_nan_mask
def ta_STOCHRSIk(x, timeperiod):
    return apply_column(x, _stochrsi, timeperiod=timeperiod, fastk_period=5, fastd_period=3, fastd_matype=0, type='k')
@error_handle_and_nan_mask
def ta_STOCHRSId(x, timeperiod):
    return apply_column(x, _stochrsi, timeperiod=timeperiod, fastk_period=5, fastd_period=3, fastd_matype=0, type='d')
_extra_function_map.update({f'ta_STOCHRSIk_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_STOCHRSIk(x, w)), name=f'ta_STOCHRSIk_{w}', arity=1) for w in [10]})
_extra_function_map.update({f'ta_STOCHRSId_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_STOCHRSId(x, w)), name=f'ta_STOCHRSId_{w}', arity=1) for w in [10]})

@error_handle_and_nan_mask
def ta_TRIX(x, timeperiod):
    return apply_column(x, talib.TRIX, timeperiod)
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
@error_handle_and_nan_mask
def ta_BBANDSu(x, timeperiod, nbdevup, nbdevdn, matype):
    return apply_column(x, _bbands, timeperiod, nbdevup, nbdevdn, matype, 'u')
@error_handle_and_nan_mask
def ta_BBANDSm(x, timeperiod, nbdevup, nbdevdn, matype):
    return apply_column(x, _bbands, timeperiod, nbdevup, nbdevdn, matype, 'm')
@error_handle_and_nan_mask
def ta_BBANDSl(x, timeperiod, nbdevup, nbdevdn, matype,):
    return apply_column(x, _bbands, timeperiod, nbdevup, nbdevdn, matype, 'l')
_extra_function_map.update({
    'ta_BBANDSu_5_2_2': _Function(function=wrap_non_picklable_objects(lambda x: ta_BBANDSu(x, 5, 2, 2, 0)), name = 'ta_BBANDSu_5_2_2', arity=1),
    'ta_BBANDSm_5_2_2': _Function(function=wrap_non_picklable_objects(lambda x: ta_BBANDSm(x, 5, 2, 2, 0)), name = 'ta_BBANDSm_5_2_2', arity=1),
    'ta_BBANDSl_5_2_2': _Function(function=wrap_non_picklable_objects(lambda x: ta_BBANDSl(x, 5, 2, 2, 0)), name = 'ta_BBANDSl_5_2_2', arity=1),
})

@error_handle_and_nan_mask
def ta_DEMA(x, timeperiod):
    return apply_column(x, talib.DEMA, timeperiod)
_extra_function_map.update({f'ta_DEMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_DEMA(x, w)), name=f'ta_DEMA_{w}', arity=1) for w in ts_wins if w >=5})

@error_handle_and_nan_mask
def ta_EMA(x, timeperiod):
    return apply_column(x, talib.EMA, timeperiod)
_extra_function_map.update({f'ta_EMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_EMA(x, w)), name=f'ta_EMA_{w}', arity=1) for w in ts_wins if w >=5})

@error_handle_and_nan_mask
def ta_HTTRENDLINE(x):
    return apply_column(x, talib.HT_TRENDLINE)
_extra_function_map.update({f'ta_HTTRENDLINE': _Function(function=wrap_non_picklable_objects(ta_HTTRENDLINE), name=f'ta_HTTRENDLINE', arity=1)})

@error_handle_and_nan_mask
def ta_KAMA(x, timeperiod):
    return apply_column(x, talib.KAMA, timeperiod)
_extra_function_map.update({f'ta_KAMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_KAMA(x, w)), name=f'ta_KAMA_{w}', arity=1) for w in ts_wins if w >=5})

def _mama(x, fastlimit, slowlimit, type=''):
    mama, fama = talib.MAMA(x, fastlimit, slowlimit)
    if type == 'mama':
        return mama
    elif type == 'fama':
        return fama
@error_handle_and_nan_mask
def ta_MAMA(x, fastlimit, slowlimit):
    return apply_column(x, _mama, fastlimit, slowlimit, 'mama')
@error_handle_and_nan_mask
def ta_FAMA(x, fastlimit, slowlimit):
    return apply_column(x, _mama, fastlimit, slowlimit, 'fama')
_extra_function_map.update({
    'ta_MAMA_0.5_0.05': _Function(function=wrap_non_picklable_objects(lambda x: ta_MAMA(x, 0.5, 0.05)), name = 'ta_MAMA_0.5_0.05', arity=1),
    'ta_FAMA_0.5_0.05': _Function(function=wrap_non_picklable_objects(lambda x: ta_FAMA(x, 0.5, 0.05)), name = 'ta_FAMA_0.5_0.05', arity=1),
})

@error_handle_and_nan_mask
def ta_SMA(x, timeperiod):
    return apply_column(x, talib.SMA, timeperiod)
_extra_function_map.update({f'ta_SMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_SMA(x, w)), name=f'ta_SMA_{w}', arity=1) for w in ts_wins if w >=5})

@error_handle_and_nan_mask
def ta_T3(x, timeperiod, vfactor):
    return apply_column(x, talib.T3, timeperiod, vfactor)
_extra_function_map.update({f'ta_T3_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_T3(x, w, 0.7)), name=f'ta_T3_{w}', arity=1) for w in ts_wins if w >=5})

@error_handle_and_nan_mask
def ta_TEMA(x, timeperiod):
    return apply_column(x, talib.TEMA, timeperiod)
_extra_function_map.update({f'ta_TEMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_TEMA(x, w)), name=f'ta_TEMA_{w}', arity=1) for w in ts_wins if w >=5})

@error_handle_and_nan_mask
def ta_TRIMA(x, timeperiod):
    return apply_column(x, talib.TRIMA, timeperiod)
_extra_function_map.update({f'ta_TRIMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_TRIMA(x, w)), name=f'ta_TRIMA_{w}', arity=1) for w in ts_wins if w >=5})

@error_handle_and_nan_mask
def ta_WMA(x, timeperiod):
    return apply_column(x, talib.WMA, timeperiod)
_extra_function_map.update({f'ta_WMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_WMA(x, w)), name=f'ta_WMA_{w}', arity=1) for w in ts_wins if w >=5})

# endregion

# region ==== TA functions: Volume ====

@error_handle_and_nan_mask
def ta_OBV(x1, x2):     ## use OBV as an arity 2 func
    return apply_column_2(x1, x2, talib.OBV)
_extra_function_map.update({f'ta_OBV': _Function(function=wrap_non_picklable_objects(ta_OBV), name=f'ta_OBV', arity=2)})

@error_handle_and_nan_mask
def cszs_ta_OBV(x1, x2):     ## use OBV as an arity 2 func
    return apply_column_2(cs_zscore(x1), cs_zscore(x2), talib.OBV)
_extra_function_map.update({f'cszs_ta_OBV': _Function(function=wrap_non_picklable_objects(cszs_ta_OBV), name=f'cszs_ta_OBV', arity=2)})

@error_handle_and_nan_mask
def tszs_ta_OBV(x1, x2, w):     ## use OBV as an arity 2 func
    return apply_column_2(ts_zscore(x1, w=w), ts_zscore(x2, w=w), talib.OBV)
_extra_function_map.update({f'tszs_{w}_ta_OBV': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_ta_OBV(x1, x2, w)), name=f'tszs_{w}_ta_OBV', arity=2) for w in tszs_wins})


def _vwma(x1, x2, w):
    pv = x1 * x2
    vwma = talib.MA(pv, timeperiod=w, matype=0) / talib.MA(x2, timeperiod=w, matype=0)
    return vwma
@error_handle_and_nan_mask
def ta_VWMA(x1, x2, w=5):     ## use VWMA as an arity 2 func
    return apply_column_2(x1, x2, _vwma, w=w)
_extra_function_map.update({f'ta_VWMA_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_VWMA(x, w)), name=f'ta_VWMA_{w}', arity=2) for w in ts_wins if w >=5})

# endregion

# region ==== TA functions: Cycle ====

@error_handle_and_nan_mask
def ta_HTDCPERIOD(x):   # Hilbert Transform
    return apply_column(x, talib.HT_DCPERIOD)
_extra_function_map.update({f'ta_HTDCPERIOD': _Function(function=wrap_non_picklable_objects(ta_HTDCPERIOD), name=f'ta_HTDCPERIOD', arity=1)})

@error_handle_and_nan_mask
def ta_HTDCPHASE(x):   # Hilbert Transform
    return apply_column(x, talib.HT_DCPHASE)
_extra_function_map.update({f'ta_HTDCPHASE': _Function(function=wrap_non_picklable_objects(ta_HTDCPHASE), name=f'ta_HTDCPHASE', arity=1)})

def _ta_HTPHASOR(x, mode):
    inphase, quadrature = talib.HT_PHASOR(x)
    if mode == 'i':
        return inphase
    elif mode == 'q':
        return quadrature
@error_handle_and_nan_mask
def ta_HTPHASOR_INPHASE(x):   # Hilbert Transform
    return apply_column(x, _ta_HTPHASOR, mode='i')
_extra_function_map.update({f'ta_HTPHASOR_INPHASE': _Function(function=wrap_non_picklable_objects(ta_HTPHASOR_INPHASE), name=f'ta_HTPHASOR_INPHASE', arity=1)})
def ta_HTPHASOR_QUADRATURE(x):   # Hilbert Transform
    return apply_column(x, _ta_HTPHASOR, mode='q')
_extra_function_map.update({f'ta_HTPHASOR_QUADRATURE': _Function(function=wrap_non_picklable_objects(ta_HTPHASOR_QUADRATURE), name=f'ta_HTPHASOR_QUADRATURE', arity=1)})

def _ta_HTSINE(x, mode):
    sine, leadsine = talib.HT_SINE(x)
    if mode == 's':
        return sine
    elif mode == 'l':
        return leadsine
@error_handle_and_nan_mask
def ta_HTSINE_SINE(x):   # Hilbert Transform
    return apply_column(x, _ta_HTSINE, mode='s')
_extra_function_map.update({f'ta_HTSINE_SINE': _Function(function=wrap_non_picklable_objects(ta_HTSINE_SINE), name=f'ta_HTSINE_SINE', arity=1)})
@error_handle_and_nan_mask
def ta_HTSINE_LEADSINE(x):   # Hilbert Transform
    return apply_column(x, _ta_HTSINE, mode='l')
_extra_function_map.update({f'ta_HTSINE_LEADSINE': _Function(function=wrap_non_picklable_objects(ta_HTSINE_LEADSINE), name=f'ta_HTSINE_LEADSINE', arity=1)})

@error_handle_and_nan_mask
def ta_HTTRENDMODE(x):   # Hilbert Transform
    return apply_column(x, talib.HT_TRENDMODE)
_extra_function_map.update({f'ta_HTTRENDMODE': _Function(function=wrap_non_picklable_objects(ta_HTTRENDMODE), name=f'ta_HTTRENDMODE', arity=1)})

from pandas_ta.cycles.reflex import np_reflex
@error_handle_and_nan_mask
def ta_REFLEX(x, w=10):
    ret = apply_column(x, np_reflex, n=w, k=w, alpha=0.04, pi=3.14159, sqrt2=1.414)
    ret[:w, :] = np.nan
    return ret
_extra_function_map.update({f'ta_REFLEX_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_REFLEX(x, w)), name=f'ta_REFLEX_{w}', arity=1) for w in ts_wins if w >=10})
       

# endregion

# region ==== TA functions: Statistic ====

@error_handle_and_nan_mask
def ta_BETA(x1, x2, w):
    return apply_column_2(x1, x2, talib.BETA, timeperiod=w)
_extra_function_map.update({f'ta_BETA_{w}': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: ta_BETA(x1, x2, w)), name=f'ta_BETA_{w}', arity=2) for w in ts_wins if w >=5})

@error_handle_and_nan_mask
def ta_CORREL(x1, x2, w):   ## Pearson's Correlation Coefficient (r)
    return apply_column_2(x1, x2, talib.CORREL, timeperiod=w)
_extra_function_map.update({f'ta_CORREL_{w}': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: ta_CORREL(x1, x2, w)), name=f'ta_CORREL_{w}', arity=2) for w in ts_wins if w >=10})

@error_handle_and_nan_mask
def ta_LINEARREG(x, w=10):  ## Linear Regression
    return apply_column(x, talib.LINEARREG, timeperiod=w)
_extra_function_map.update({f'ta_LINEARREG_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_LINEARREG(x, w)), name=f'ta_LINEARREG_{w}', arity=1) for w in ts_wins if w >=10})

@error_handle_and_nan_mask
def ta_LINEARREG_ANGLE(x, w=10):  ## Linear Regression Angle
    return apply_column(x, talib.LINEARREG_ANGLE, timeperiod=w)
_extra_function_map.update({f'ta_LINEARREG_ANGLE_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_LINEARREG_ANGLE(x, w)), name=f'ta_LINEARREG_ANGLE_{w}', arity=1) for w in ts_wins if w >=10})

@error_handle_and_nan_mask
def ta_LINEARREG_INTERCEPT(x, w=10):  ## Linear Regression Intercept
    return apply_column(x, talib.LINEARREG_INTERCEPT, timeperiod=w)
_extra_function_map.update({f'ta_LINEARREG_INTERCEPT_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_LINEARREG_INTERCEPT(x, w)), name=f'ta_LINEARREG_INTERCEPT_{w}', arity=1) for w in ts_wins if w >=10})

@error_handle_and_nan_mask
def ta_LINEARREG_SLOPE(x, w=10):  ## Linear Regression Slope
    return apply_column(x, talib.LINEARREG_SLOPE, timeperiod=w)
_extra_function_map.update({f'ta_LINEARREG_SLOPE_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_LINEARREG_SLOPE(x, w)), name=f'ta_LINEARREG_SLOPE_{w}', arity=1) for w in ts_wins if w >=10})

@error_handle_and_nan_mask
def ta_TSF(x, w=10):  ## Time Series Forecast
    return apply_column(x, talib.TSF, timeperiod=w)
_extra_function_map.update({f'ta_TSF_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_TSF(x, w)), name=f'ta_TSF_{w}', arity=1) for w in ts_wins if w >=10})

def _mad(x, axis):
    return np.mean(
        np.fabs(
            x - np.mean(x, axis=axis).reshape(-1, 1)
            ), 
        axis=1)
@error_handle_and_nan_mask
def ta_MAD(x, w=5):
    return np_rolling_apply(x, w, _mad)
_extra_function_map.update({f'ta_MAD_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_MAD(x, w)), name=f'ta_MAD_{w}', arity=1) for w in ts_wins if w >=5})


# endregion

# region ==== TA functions: Math Transform ====

@error_handle_and_nan_mask
def ta_ACOS(x):  
    return apply_column(x, talib.ACOS)
_extra_function_map.update({f'ta_ACOS': _Function(function=wrap_non_picklable_objects(ta_ACOS), name=f'ta_ACOS', arity=1)})

@error_handle_and_nan_mask
def ta_ASIN(x):  
    return apply_column(x, talib.ASIN)
_extra_function_map.update({f'ta_ASIN': _Function(function=wrap_non_picklable_objects(ta_ASIN), name=f'ta_ASIN', arity=1)})

@error_handle_and_nan_mask
def ta_ATAN(x):  
    return apply_column(x, talib.ATAN)
_extra_function_map.update({f'ta_ATAN': _Function(function=wrap_non_picklable_objects(ta_ATAN), name=f'ta_ATAN', arity=1)})

@error_handle_and_nan_mask
def ta_CEIL(x):  
    return apply_column(x, talib.CEIL)
_extra_function_map.update({f'ta_CEIL': _Function(function=wrap_non_picklable_objects(ta_CEIL), name=f'ta_CEIL', arity=1)})

@error_handle_and_nan_mask
def ta_COS(x):  
    return apply_column(x, talib.COS)
_extra_function_map.update({f'ta_COS': _Function(function=wrap_non_picklable_objects(ta_COS), name=f'ta_COS', arity=1)})

@error_handle_and_nan_mask
def ta_COSH(x):  
    return apply_column(x, talib.COSH)
_extra_function_map.update({f'ta_COSH': _Function(function=wrap_non_picklable_objects(ta_COSH), name=f'ta_COSH', arity=1)})

@error_handle_and_nan_mask
def ta_FLOOR(x):  
    return apply_column(x, talib.FLOOR)
_extra_function_map.update({f'ta_FLOOR': _Function(function=wrap_non_picklable_objects(ta_FLOOR), name=f'ta_FLOOR', arity=1)})

@error_handle_and_nan_mask
def ta_LN(x):  
    return apply_column(x, talib.LN)
_extra_function_map.update({f'ta_LN': _Function(function=wrap_non_picklable_objects(ta_LN), name=f'ta_LN', arity=1)})

@error_handle_and_nan_mask
def ta_SIN(x):  
    return apply_column(x, talib.SIN)
_extra_function_map.update({f'ta_SIN': _Function(function=wrap_non_picklable_objects(ta_SIN), name=f'ta_SIN', arity=1)})

@error_handle_and_nan_mask
def ta_SINH(x):  
    return apply_column(x, talib.SINH)
_extra_function_map.update({f'ta_SINH': _Function(function=wrap_non_picklable_objects(ta_SINH), name=f'ta_SINH', arity=1)})

@error_handle_and_nan_mask
def ta_TAN(x):  
    return apply_column(x, talib.TAN)
_extra_function_map.update({f'ta_TAN': _Function(function=wrap_non_picklable_objects(ta_TAN), name=f'ta_TAN', arity=1)})

@error_handle_and_nan_mask
def ta_TANH(x):  
    return apply_column(x, talib.TANH)
_extra_function_map.update({f'ta_TANH': _Function(function=wrap_non_picklable_objects(ta_TANH), name=f'ta_TANH', arity=1)})

# endregion

# endregion TA


# region ==== Basic ====

# region ==== Basic: arity > 1, w/ & w/o ts zscore ====

@error_handle_and_nan_mask
def add(x1, x2):
    return np.add(x1, x2)
_extra_function_map.update({f'add': _Function(function=wrap_non_picklable_objects(add), name=f'add', arity=2)})

@error_handle_and_nan_mask
def cszs_add(x1, x2):
    return np.add(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_add': _Function(function=wrap_non_picklable_objects(cszs_add), name=f'cszs_add', arity=2)})

@error_handle_and_nan_mask
def tszs_add(x1, x2, w):
    return np.add(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_add': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_add(x1, x2, w)), name=f'tszs_{w}_add', arity=2) for w in tszs_wins})


@error_handle_and_nan_mask
def sub(x1, x2):
    return np.subtract(x1, x2)
_extra_function_map.update({f'sub': _Function(function=wrap_non_picklable_objects(sub), name=f'sub', arity=2)})

@error_handle_and_nan_mask
def cszs_sub(x1, x2):
    return np.subtract(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_sub': _Function(function=wrap_non_picklable_objects(cszs_sub), name=f'cszs_sub', arity=2)})

@error_handle_and_nan_mask
def tszs_sub(x1, x2, w):
    return np.subtract(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_sub': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_sub(x1, x2, w)), name=f'tszs_{w}_sub', arity=2) for w in tszs_wins})


@error_handle_and_nan_mask
def mul(x1, x2):
    return np.multiply(x1, x2)
_extra_function_map.update({f'mul': _Function(function=wrap_non_picklable_objects(mul), name=f'mul', arity=2)})

@error_handle_and_nan_mask
def cszs_mul(x1, x2):
    return np.multiply(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_mul': _Function(function=wrap_non_picklable_objects(cszs_mul), name=f'cszs_mul', arity=2)})

@error_handle_and_nan_mask
def tszs_mul(x1, x2, w):
    return np.multiply(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_mul': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_mul(x1, x2, w)), name=f'tszs_{w}_mul', arity=2) for w in tszs_wins})


@error_handle_and_nan_mask
def div(x1, x2):
    return np.where(x2 > 0.001, np.divide(x1, x2), 1.)
_extra_function_map.update({f'div': _Function(function=wrap_non_picklable_objects(div), name=f'div', arity=2)})

@error_handle_and_nan_mask
def cszs_div(x1, x2):
    cszs_x2 = cs_zscore(x2)
    return np.where(cszs_x2 > 0.001, np.divide(cs_zscore(x1), cszs_x2), 1.)
_extra_function_map.update({f'cszs_div': _Function(function=wrap_non_picklable_objects(cszs_div), name=f'cszs_div', arity=2)})

@error_handle_and_nan_mask
def tszs_div(x1, x2, w):
    tszs_x1 = ts_zscore(x1, w=w)
    tszs_x2 = ts_zscore(x2, w=w)
    return np.where(np.abs(tszs_x2) > 0.001, np.divide(tszs_x1, tszs_x2), 1.)
_extra_function_map.update({f'tszs_{w}_div': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_div(x1, x2, w)), name=f'tszs_{w}_div', arity=2) for w in tszs_wins})


@error_handle_and_nan_mask
def max(x1, x2):
    return np.maximum(x1, x2)
_extra_function_map.update({f'max': _Function(function=wrap_non_picklable_objects(max), name=f'max', arity=2)})

@error_handle_and_nan_mask
def cszs_max(x1, x2):
    return np.maximum(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_max': _Function(function=wrap_non_picklable_objects(cszs_max), name=f'cszs_max', arity=2)})

@error_handle_and_nan_mask
def tszs_max(x1, x2, w):
    return np.maximum(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_max': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_max(x1, x2, w)), name=f'tszs_{w}_max', arity=2) for w in tszs_wins})


@error_handle_and_nan_mask
def min(x1, x2):
    return np.minimum(x1, x2)
_extra_function_map.update({f'min': _Function(function=wrap_non_picklable_objects(min), name=f'min', arity=2)})

@error_handle_and_nan_mask
def cszs_min(x1, x2):
    return np.minimum(cs_zscore(x1), cs_zscore(x2))
_extra_function_map.update({f'cszs_min': _Function(function=wrap_non_picklable_objects(cszs_min), name=f'cszs_min', arity=2)})

@error_handle_and_nan_mask
def tszs_min(x1, x2, w):
    return np.minimum(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_min': _Function(function=wrap_non_picklable_objects(lambda x1, x2, w=w: tszs_min(x1, x2, w)), name=f'tszs_{w}_min', arity=2) for w in tszs_wins})

# endregion

# region ==== Basic: arity = 1, simple ====

@error_handle_and_nan_mask
def sqrt(x):
    return np.sqrt(np.abs(x))
_extra_function_map.update({f'sqrt': _Function(function=wrap_non_picklable_objects(sqrt), name=f'sqrt', arity=1)})

@error_handle_and_nan_mask
def signed_sqrt(x):
    return np.sqrt(np.abs(x)) * np.sign(x)
_extra_function_map.update({f'signed_sqrt': _Function(function=wrap_non_picklable_objects(signed_sqrt), name=f'signed_sqrt', arity=1)})

@error_handle_and_nan_mask
def log(x):
    return np.where(np.abs(x) > 0.0001, np.log(np.abs(x)), 0.)
_extra_function_map.update({f'log': _Function(function=wrap_non_picklable_objects(log), name=f'log', arity=1)})

@error_handle_and_nan_mask
def signed_log(x):
    return np.log(np.abs(x)) * np.sign(x)
_extra_function_map.update({f'signed_log': _Function(function=wrap_non_picklable_objects(signed_log), name=f'signed_log', arity=1)})

@error_handle_and_nan_mask
def neg(x):
    return np.negative(x)
_extra_function_map.update({f'neg': _Function(function=wrap_non_picklable_objects(neg), name=f'neg', arity=1)})

@error_handle_and_nan_mask
def inv(x):
    return np.where(np.abs(x) > 0.0001, 1. / x, 0.)
_extra_function_map.update({f'inv': _Function(function=wrap_non_picklable_objects(inv), name=f'inv', arity=1)})

@error_handle_and_nan_mask
def abs(x):
    return np.abs(x)
_extra_function_map.update({f'abs': _Function(function=wrap_non_picklable_objects(abs), name=f'abs', arity=1)})

@error_handle_and_nan_mask
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
_extra_function_map.update({f'sigmoid': _Function(function=wrap_non_picklable_objects(sigmoid), name=f'sigmoid', arity=1)})

@error_handle_and_nan_mask
def pow2(x):
    return np.power(x, 2)
_extra_function_map.update({f'pow2': _Function(function=wrap_non_picklable_objects(pow2), name=f'pow2', arity=1)})

# endregion

# endregion Basic


# region ==== Multi-conditions ====

@error_handle_and_nan_mask
def clear_by_cond(x1, x2, x3):
    """if x1 < x2 (keep NaN if and only if both x1 and x2 are NaN), then 0, else x3"""
    return np.where(x1 < x2, 0, np.where(~np.isnan(x1) | ~np.isnan(x2), x3, np.nan))
_extra_function_map.update({'clear_by_cond': _Function(function=wrap_non_picklable_objects(clear_by_cond), name="clear_by_cond", arity=3)})

error_handle_and_nan_mask
def if_then_else(x1, x2, x3):
    """if x1 is nonzero (keep NaN), then x2, else x3"""
    return np.where(x1, x2, np.where(~np.isnan(x1), x3, np.nan))
_extra_function_map.update({'if_then_else': _Function(function=wrap_non_picklable_objects(if_then_else), name="if_then_else", arity=3)})

@error_handle_and_nan_mask
def if_cond_then_else(x1, x2, x3, x4):
    """if x1 < x2 (keep NaN if and only if both x1 and x2 are NaN), then x3, else x4"""
    return np.where(x1 < x2, x3, np.where(~np.isnan(x1) | ~np.isnan(x2), x4, np.nan))
_extra_function_map.update({'if_cond_then_else': _Function(function=wrap_non_picklable_objects(if_cond_then_else), name="if_cond_then_else", arity=4)})

# endregion

# TODO: pure pandas-ta funcs, inverse cv...