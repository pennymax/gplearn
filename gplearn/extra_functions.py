import copy

import numpy as np
import pandas as pd

from .functions import _Function
from joblib import wrap_non_picklable_objects

def error_state_decorator(func):
    def wrapper(A, *args, **kwargs):
        with np.errstate(over='ignore', under='ignore'):
            return func(A, *args, **kwargs)
    return wrapper

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
    delta = ret.diff(window)
    shifted_ret = ret.shift(window)
    factor = np.where(shifted_ret == 0, np.nan, (delta / shifted_ret.abs()) * np.sign(shifted_ret))
    return factor

@error_state_decorator
def rolling_nanmean(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window).mean().to_numpy(dtype=np.double)
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
def rolling_correlation(A, B, window=5):
    ret1 = pd.DataFrame(A)
    ret2 = pd.DataFrame(B)
    factor = ret1.rolling(window).corr(ret2).to_numpy(dtype=np.double)
    return factor

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

rolling_windows = [1, 3, 5, 10, 20, 40]
# ts mean
_extra_function_map.update({f'ts_mean_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_nanmean(x, w)), name=f'ts_mean_{w}', arity=1) for w in rolling_windows if w > 1})
# ts max
_extra_function_map.update({f'ts_max_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_max(x, w)), name=f'ts_max_{w}', arity=1) for w in rolling_windows if w > 1})
# ts min
_extra_function_map.update({f'ts_min_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_min(x, w)), name=f'ts_min_{w}', arity=1) for w in rolling_windows if w > 1})
# ts std
_extra_function_map.update({f'ts_std_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_nanstd(x, w)), name=f'ts_std_{w}', arity=1) for w in rolling_windows if w >=5})
# ts skew
_extra_function_map.update({f'ts_skew_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_skew(x, w)), name=f'ts_skew_{w}', arity=1) for w in rolling_windows if w >=5})
# ts kurt
_extra_function_map.update({f'ts_kurt_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_kurt(x, w)), name=f'ts_kurt_{w}', arity=1) for w in rolling_windows if w >=5})
# ts corr
_extra_function_map.update({f'ts_correlation_{w}': _Function(function=wrap_non_picklable_objects(lambda x, y, w=w: rolling_correlation(x, y, w)), name=f'ts_correlation_{w}', arity=2) for w in rolling_windows if w >=3})
# ts argmin
_extra_function_map.update({f'ts_argmin_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_argmin(x, w)), name=f'ts_argmin_{w}', arity=1) for w in rolling_windows if w >=3})
# ts argmax
_extra_function_map.update({f'ts_argmax_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_argmax(x, w)), name=f'ts_argmax_{w}', arity=1) for w in rolling_windows if w >=3})
# ts rank
_extra_function_map.update({f'ts_rank_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_rank(x, w)), name=f'ts_rank_{w}', arity=1) for w in rolling_windows if w >=3})
# ts delay
_extra_function_map.update({f'delay_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: delay(x, w)), name=f'delay_{w}', arity=1) for w in rolling_windows})
# ts delta
_extra_function_map.update({f'delta_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: delta(x, w)), name=f'delta_{w}', arity=1) for w in rolling_windows})
# ts delta pct
_extra_function_map.update({f'delta_pct_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: delta_pct(x, w)), name=f'delta_pct_{w}', arity=1) for w in rolling_windows})
# ts sum
_extra_function_map.update({f'ts_sum_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: rolling_sum(x, w)), name=f'ts_sum_{w}', arity=1) for w in rolling_windows if w > 1})

