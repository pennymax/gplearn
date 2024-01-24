import copy

import numpy as np
import pandas as pd

from .functions import _Function, _protected_division
from joblib import wrap_non_picklable_objects
import pandas_ta as ta

def error_state_decorator(func):
    def wrapper(A, *args, **kwargs):
        with np.errstate(over='ignore', under='ignore'):
            return func(A, *args, **kwargs)
    return wrapper

@error_state_decorator
def clear_by_cond(x1, x2, x3):
    """if x1 < x2 (keep NaN if and only if both x1 and x2 are NaN), then 0, else x3"""
    return np.where(x1 < x2, 0, np.where(~np.isnan(x1) | ~np.isnan(x2), x3, np.nan))

error_state_decorator
def if_then_else(x1, x2, x3):
    """if x1 is nonzero (keep NaN), then x2, else x3"""
    return np.where(x1, x2, np.where(~np.isnan(x1), x3, np.nan))

@error_state_decorator
def if_cond_then_else(x1, x2, x3, x4):
    """if x1 < x2 (keep NaN if and only if both x1 and x2 are NaN), then x3, else x4"""
    return np.where(x1 < x2, x3, np.where(~np.isnan(x1) | ~np.isnan(x2), x4, np.nan))

@error_state_decorator
def scale(A, scaler=1):
    ret = pd.DataFrame(A)
    factor = scaler * ret.div(ret.sum(axis=1), axis=0)
    factor = factor.replace([np.inf, -np.inf], 0).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rank(A):
    ret = pd.DataFrame(A)
    factor = ret.rank(axis=1).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def delay(A, window=1):
    ret = pd.DataFrame(A)
    factor = ret.shift(window).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def delta(A, window=1):
    ret = pd.DataFrame(A)
    factor = ret.diff(window).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def delta_pct(A, window):
    ret = pd.DataFrame(A)
    shifted_ret = ret.shift(window)
    factor = np.where(shifted_ret == 0, np.nan, ret.diff(window) / shifted_ret.abs() - 1)
    return factor

@error_state_decorator
def rolling_delta1_pct_mean(A, window):
    ret = pd.DataFrame(A)
    shifted_ret = ret.shift(window)
    delta_pct = np.where(shifted_ret == 0, np.nan, ret.diff(1) / shifted_ret.abs() - 1)
    factor = pd.DataFrame(delta_pct).rolling(window).mean().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_nanmean(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).mean().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_median(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).median().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_nanstd(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).std().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_max(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).max()
    factor = factor.to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_min(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).min()
    factor = factor.to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_midpoint(A, window=5):
    ret = pd.DataFrame(A)
    min = ret.rolling(window).min()
    max = ret.rolling(window).max()
    factor = ((min + max) / 2).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_corr(A, B, window=5):
    """moving covariance of x1 and x2"""
    ret1 = pd.DataFrame(A)
    ret2 = pd.DataFrame(B)
    factor = ret1.rolling(window).corr(ret2).replace([np.inf, -np.inf], np.nan)
    return factor.to_numpy(dtype=np.double)

@error_state_decorator
def rolling_autocorr(A, window=5, lag=1):
    """moving autocorrelation coefficient between x and x lag i period"""
    ret = pd.DataFrame(A)
    ret_lag = ret.shift(lag)
    factor = ret.rolling(window).corr(ret_lag).replace([np.inf, -np.inf], np.nan)
    return factor.to_numpy(dtype=np.double)

@error_state_decorator
def rolling_argmin(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).apply(lambda x: np.argmin(x), raw=True).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_argmax(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).apply(lambda x: np.argmax(x), raw=True).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_argmaxmin(A, window=5):
    """relative position of maximum x1 to minimum x1 in the last d datetimes"""
    return rolling_argmax(A, window) - rolling_argmin(A, window)

@error_state_decorator
def rolling_rank(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).rank().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def pow(A, pow=2):
    return np.power(A,pow)

@error_state_decorator
def rolling_skew(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).skew().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_kurt(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).kurt().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_sum(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).sum().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_inverse_cv(A, window=5):
    """moving inverse of coefficient of variance"""
    std = rolling_nanstd(A, window)
    mean = rolling_nanmean(A, window)
    cv = _protected_division(mean, std)
    return cv

@error_state_decorator
def rolling_regression_beta(A, B, window=5):
    """beta of regression x1 onto x2 in the last d datetimes"""
    cov = rolling_corr(A, B, window)
    std = rolling_nanstd(B, window)
    factor = _protected_division(cov, std ** 2)
    return factor

@error_state_decorator
def rolling_linear_slope(A, window=5):
    """beta of regression x1 in the last d datetimes onto (1, 2, ..., d)"""
    num_rows, num_cols = A.shape
    B = np.tile(np.arange(1, num_rows + 1).reshape(-1, 1), num_cols)
    factor = _protected_division(rolling_corr(A, B, window), rolling_nanstd(B, window) ** 2)
    return factor

@error_state_decorator
def rolling_linear_intercept(A, window=5):
    """intercept of regression x1 in the last d datetimes onto (1, 2, ..., d)"""
    factor = rolling_nanmean(A, window) - (1 + window) / 2 * rolling_linear_slope(A, window)
    return factor

@error_state_decorator
def rolling_ema(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.ema(x, length=window)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_dema(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.dema(x, length=window)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_wma(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.wma(x, length=window)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_kama(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.kama(x, length=window)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_entropy(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.entropy(x, length=window)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_quantile_25(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.quantile(x, length=window, q=0.25)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_quantile_75(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.quantile(x, length=window, q=0.75)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_lingreg_slope(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.linreg(x, length=window, slope=True)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_lingreg_intercept(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.linreg(x, length=window, intercept=True)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_lingreg_corr(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.linreg(x, length=window, r=True)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_decay(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.decay(x, length=window, mode='exp')).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_decreasing(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.decreasing(x, length=window, asint=True)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_increasing(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.increasing(x, length=window, asint=True)).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_vhf(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.apply(lambda x: ta.vhf(x, length=window)).to_numpy(dtype=np.double)
    return factor


## Build extra function map

_extra_function_map = {
    'rank': _Function(function=wrap_non_picklable_objects(lambda x: rank(x)), name = 'rank', arity=1),
    'scale_1': _Function(function=wrap_non_picklable_objects(lambda x: scale(x, 1)), name = 'scale_1', arity=1),
    'power_2': _Function(function=wrap_non_picklable_objects(lambda x: pow(x, 2)), name = 'power_2', arity=1),
    'power_3': _Function(function=wrap_non_picklable_objects(lambda x: pow(x, 3)), name = 'power_3', arity=1),

    'clear_by_cond': _Function(function=wrap_non_picklable_objects(clear_by_cond), name="clear_by_cond", arity=3),
    'if_then_else': _Function(function=wrap_non_picklable_objects(if_then_else), name="if_then_else", arity=3),
    'if_cond_then_else': _Function(function=wrap_non_picklable_objects(if_cond_then_else), name="if_cond_then_else", arity=4),
}

rolling_windows = [1, 3, 5, 10, 20, 40, 60]
lag_range = [1, 3, 5, 10] # used in autocorr and tbd
# ts mean
_extra_function_map.update({f'ts_mean_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_nanmean(x, w)), name=f'ts_mean_{w}', arity=1) for w in rolling_windows if w > 1})
# ts median
_extra_function_map.update({f'ts_median_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_median(x, w)), name=f'ts_median_{w}', arity=1) for w in rolling_windows if w > 2})
# ts max
_extra_function_map.update({f'ts_max_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_max(x, w)), name=f'ts_max_{w}', arity=1) for w in rolling_windows if w > 1})
# ts min
_extra_function_map.update({f'ts_min_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_min(x, w)), name=f'ts_min_{w}', arity=1) for w in rolling_windows if w > 1})
# ts midpoint
_extra_function_map.update({f'ts_midpoint_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_midpoint(x, w)), name=f'ts_midpoint_{w}', arity=1) for w in rolling_windows if w > 1})
# ts std
_extra_function_map.update({f'ts_std_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_nanstd(x, w)), name=f'ts_std_{w}', arity=1) for w in rolling_windows if w >=5})
# ts skew
_extra_function_map.update({f'ts_skew_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_skew(x, w)), name=f'ts_skew_{w}', arity=1) for w in rolling_windows if w >=5})
# ts kurt
_extra_function_map.update({f'ts_kurt_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_kurt(x, w)), name=f'ts_kurt_{w}', arity=1) for w in rolling_windows if w >=5})
# ts inverse cv (= mean / std)
_extra_function_map.update({f'ts_inverse_cv_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_inverse_cv(x, w)), name=f'ts_inverse_cv_{w}', arity=1) for w in rolling_windows if w >=5})
# ts corr
_extra_function_map.update({f'ts_correlation_{w}': _Function(function=wrap_non_picklable_objects(lambda x, y, w=w: rolling_corr(x, y, w)), name=f'ts_correlation_{w}', arity=2) for w in rolling_windows if w >=3})
# ts autocorr
_extra_function_map.update({f'ts_autocorr_{w}_{l}': _Function(function=wrap_non_picklable_objects(lambda x, w=w, l=l: rolling_autocorr(x, w, l)), name=f'ts_autocorr_{w}_{l}', arity=1) for w in rolling_windows for l in lag_range if w >= 10})
# ts argmin
_extra_function_map.update({f'ts_argmin_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_argmin(x, w)), name=f'ts_argmin_{w}', arity=1) for w in rolling_windows if w >=3})
# ts argmax
_extra_function_map.update({f'ts_argmax_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_argmax(x, w)), name=f'ts_argmax_{w}', arity=1) for w in rolling_windows if w >=3})
# ts argmaxmin
_extra_function_map.update({f'ts_argmaxmin_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_argmaxmin(x, w)), name=f'ts_argmaxmin_{w}', arity=1) for w in rolling_windows if w >=3})
# ts rank
_extra_function_map.update({f'ts_rank_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_rank(x, w)), name=f'ts_rank_{w}', arity=1) for w in rolling_windows if w >=3})
# ts delay
_extra_function_map.update({f'delay_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: delay(x, w)), name=f'delay_{w}', arity=1) for w in rolling_windows})
# ts delta
_extra_function_map.update({f'delta_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: delta(x, w)), name=f'delta_{w}', arity=1) for w in rolling_windows})
# ts delta pct
_extra_function_map.update({f'delta_pct_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: delta_pct(x, w)), name=f'delta_pct_{w}', arity=1) for w in rolling_windows})
# ts delta1 pct mean
_extra_function_map.update({f'ts_delta1pct_mean_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_delta1_pct_mean(x, w)), name=f'ts_delta1pct_mean_{w}', arity=1) for w in rolling_windows if w > 1})
# ts sum
_extra_function_map.update({f'ts_sum_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_sum(x, w)), name=f'ts_sum_{w}', arity=1) for w in rolling_windows if w > 1})
# ts ema
_extra_function_map.update({f'ts_ema_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_ema(x, w)), name=f'ts_ema_{w}', arity=1) for w in rolling_windows if w >=5})
# ts dema
_extra_function_map.update({f'ts_dema_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_dema(x, w)), name=f'ts_dema_{w}', arity=1) for w in rolling_windows if w >=5})
# ts wma
_extra_function_map.update({f'ts_wma_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_wma(x, w)), name=f'ts_wma_{w}', arity=1) for w in rolling_windows if w >=5})
# ts kama
_extra_function_map.update({f'ts_kama_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_kama(x, w)), name=f'ts_kama_{w}', arity=1) for w in rolling_windows if w >=5})
# ts quantile 0.25 (bottom, <25% part)
_extra_function_map.update({f'ts_quantile25_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_quantile_25(x, w)), name=f'ts_quantile25_{w}', arity=1) for w in rolling_windows if w >=5})
# ts quantile 0.25 (top, >75% part)
_extra_function_map.update({f'ts_quantile75_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_quantile_75(x, w)), name=f'ts_quantile75_{w}', arity=1) for w in rolling_windows if w >=5})
# ts linreg slope (x over index [1, window+1])
_extra_function_map.update({f'ts_linreg_slope_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_lingreg_slope(x, w)), name=f'ts_linreg_slope_{w}', arity=1) for w in rolling_windows if w >=10})
# ts linreg intercept (x over index [1, window+1])
_extra_function_map.update({f'ts_linreg_intercept_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_lingreg_intercept(x, w)), name=f'ts_linreg_intercept_{w}', arity=1) for w in rolling_windows if w >=10})
# ts linreg correlation (x over index [1, window+1])
_extra_function_map.update({f'ts_linreg_corr_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_lingreg_corr(x, w)), name=f'ts_linreg_corr_{w}', arity=1) for w in rolling_windows if w >=10})
# ts decay
_extra_function_map.update({f'ts_decay_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_decay(x, w)), name=f'ts_decay_{w}', arity=1) for w in rolling_windows if w >=5})
# ts decreasing
_extra_function_map.update({f'ts_decreasing_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_decreasing(x, w)), name=f'ts_decreasing_{w}', arity=1) for w in rolling_windows if w >=3})
# ts increasing
_extra_function_map.update({f'ts_increasing_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_increasing(x, w)), name=f'ts_increasing_{w}', arity=1) for w in rolling_windows if w >=3})
# ts vhf
_extra_function_map.update({f'ts_vhf_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_vhf(x, w)), name=f'ts_vhf_{w}', arity=1) for w in rolling_windows if w >=5})


############
## disabled
############

# ts entropy (need to have all ts data positive, disable now)
# _extra_function_map.update({f'ts_entropy_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_entropy(x, w)), name=f'ts_entropy_{w}', arity=1) for w in rolling_windows if w >=10})
# ts regression beta (x over y)
# _extra_function_map.update({f'ts_regr_beta_{w}': _Function(function=wrap_non_picklable_objects(lambda x, y, w=w: rolling_regression_beta(x, y, w)), name=f'ts_regr_beta_{w}', arity=2) for w in rolling_windows if w >=3})
# ts regression beta (x over fixed index [1, window+1])
# _extra_function_map.update({f'ts_lin_beta_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_linear_slope(x, w)), name=f'ts_lin_beta_{w}', arity=1) for w in rolling_windows if w >=3})
# ts regression intercept (x over fixed index [1, window+1])
# _extra_function_map.update({f'ts_lin_intercept_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_linear_intercept(x, w)), name=f'ts_lin_intercept_{w}', arity=1) for w in rolling_windows if w >=3})

