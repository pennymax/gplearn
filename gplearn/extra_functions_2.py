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

# def error_state_decorator(func):
#     def wrapper(A, *args, **kwargs):
#         with np.errstate(over='ignore', under='ignore'):
#             return func(A, *args, **kwargs)
#     return wrapper

def apply_column(x, func, *args, **kwargs):
    r = np.empty_like(x)
    for i in range(x.shape[1]):
        r[:, i] = func(x[:, i], *args, **kwargs)
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
    return np.where(shifted_ret == 0, np.nan, ret.diff(w) / shifted_ret.abs() - 1)
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
def ts_max(x, w=5):
    return pd.DataFrame(x).rolling(w).max().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_max_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_max(x, w)), name=f'ts_max_{w}', arity=1) for w in ts_wins if w > 1})

@error_handle_and_nan_mask
def ts_min(x, w=5):
    return pd.DataFrame(x).rolling(w).min().to_numpy(dtype=np.double)
_extra_function_map.update({f'ts_min_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_min(x, w)), name=f'ts_min_{w}', arity=1) for w in ts_wins if w > 1})

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
_extra_function_map.update({'ta_APO_12_26': _Function(function=wrap_non_picklable_objects(lambda x: ta_APO(x, 12, 26, 0)), name = 'ta_APO_12_26', arity=1)})

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
_extra_function_map.update({'ta_PPO_12_26': _Function(function=wrap_non_picklable_objects(lambda x: ta_PPO(x, 12, 26, 0)), name = 'ta_PPO_12_26', arity=1)})

@error_handle_and_nan_mask
def ta_RSI(x, timeperiod):
    return apply_column(x, talib.RSI, timeperiod)
_extra_function_map.update({f'ta_RSI_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ta_RSI(x, w)), name=f'ta_RSI_{w}', arity=1) for w in [10]})

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

# endregion TA


# region ==== Basic ====

# region ==== Basic: arity > 1, need take ts zscore ====

@error_handle_and_nan_mask
def tszs_add(x1, x2, w=60):
    return np.add(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_add': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_add(x, w)), name=f'tszs_{w}_add', arity=2) for w in tszs_wins})

@error_handle_and_nan_mask
def tszs_sub(x1, x2, w=60):
    return np.subtract(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_sub': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_sub(x, w)), name=f'tszs_{w}_sub', arity=2) for w in tszs_wins})

@error_handle_and_nan_mask
def tszs_mul(x1, x2, w=60):
    return np.multiply(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_mul': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_mul(x, w)), name=f'tszs_{w}_mul', arity=2) for w in tszs_wins})

@error_handle_and_nan_mask
def tszs_div(x1, x2, w=60):
    tszs_x1 = ts_zscore(x1, w=w)
    tszs_x2 = ts_zscore(x2, w=w)
    return np.where(np.abs(tszs_x2) > 0.001, np.divide(tszs_x1, tszs_x2), 1.)
_extra_function_map.update({f'tszs_{w}_div': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_div(x, w)), name=f'tszs_{w}_div', arity=2) for w in tszs_wins})

@error_handle_and_nan_mask
def tszs_max(x1, x2, w=60):
    return np.maximum(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_max': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_max(x, w)), name=f'tszs_{w}_max', arity=2) for w in tszs_wins})

@error_handle_and_nan_mask
def tszs_min(x1, x2, w=60):
    return np.minimum(ts_zscore(x1, w=w), ts_zscore(x2, w=w))
_extra_function_map.update({f'tszs_{w}_min': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_min(x, w)), name=f'tszs_{w}_min', arity=2) for w in tszs_wins})

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

# endregion

# endregion Basic


