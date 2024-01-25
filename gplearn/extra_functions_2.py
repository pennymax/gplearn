import numpy as np
import pandas as pd

from .functions import _Function, _protected_division
from joblib import wrap_non_picklable_objects
import pandas_ta as ta
import talib

## better to create feature in upstream
## [mom] bop, cci



def error_state_decorator(func):
    def wrapper(A, *args, **kwargs):
        with np.errstate(over='ignore', under='ignore'):
            return func(A, *args, **kwargs)
    return wrapper

def apply_column(x, func, *args, **kwargs):
    r = np.empty_like(x)
    for i in range(x.shape[1]):
        r[:, i] = func(x[:, i], *args, **kwargs)
    return r


@error_state_decorator
def ta_APO(x, fast, slow):
    return apply_column(x, talib.APO, fast, slow)

def _macd(x, fast, slow, signal, type=''):
    m, s, h = talib.MACD(x, fast, slow, signal)
    if type == 's':
        return s
    elif type == 'h':
        return h
    else:
        return m

@error_state_decorator
def ta_MACD(x, fast, slow, signal):
    return apply_column(x, _macd, fast, slow, signal, '')
@error_state_decorator
def ta_MACDs(x, fast, slow, signal):
    return apply_column(x, _macd, fast, slow, signal, 's')
@error_state_decorator
def ta_MACDh(x, fast, slow, signal):
    return apply_column(x, _macd, fast, slow, signal, 'h')



############ Below are all arity > 1, need to do ts zscore! ############

@error_state_decorator
def ts_zscore(x, w=60):
    df = pd.DataFrame(x)
    r = df.rolling(window=w, min_periods=1)
    return ((df - r.mean()) / r.std()).to_numpy(np.double)

@error_state_decorator
def tszs_add(x1, x2, w=60):
    return np.add(ts_zscore(x1, w=w), ts_zscore(x2, w=w))

@error_state_decorator
def tszs_sub(x1, x2, w=60):
    return np.subtract(ts_zscore(x1, w=w), ts_zscore(x2, w=w))

@error_state_decorator
def tszs_mul(x1, x2, w=60):
    return np.multiply(ts_zscore(x1, w=w), ts_zscore(x2, w=w))

@error_state_decorator
def tszs_div(x1, x2, w=60):
    tszs_x1 = ts_zscore(x1, w=w)
    tszs_x2 = ts_zscore(x2, w=w)
    return np.where(np.abs(tszs_x2) > 0.001, np.divide(tszs_x1, tszs_x2), 1.)

@error_state_decorator
def tszs_max(x1, x2, w=60):
    return np.maximum(ts_zscore(x1, w=w), ts_zscore(x2, w=w))

@error_state_decorator
def tszs_min(x1, x2, w=60):
    return np.minimum(ts_zscore(x1, w=w), ts_zscore(x2, w=w))




############ Construct function map ############

_extra_function_map = {
    'ta_APO_12_26': _Function(function=wrap_non_picklable_objects(lambda x: ta_APO(x, 12, 26)), name = 'ta_APO_12_26', arity=1),
    'ta_MACD_12_26_9': _Function(function=wrap_non_picklable_objects(lambda x: ta_MACD(x, 12, 26, 9)), name = 'ta_MACD_12_26_9', arity=1),
    'ta_MACDs_12_26_9': _Function(function=wrap_non_picklable_objects(lambda x: ta_MACDs(x, 12, 26, 9)), name = 'ta_MACDs_12_26_9', arity=1),
    'ta_MACDh_12_26_9': _Function(function=wrap_non_picklable_objects(lambda x: ta_MACDh(x, 12, 26, 9)), name = 'ta_MACDh_12_26_9', arity=1),
}

tszs_win = [60, 120]
_extra_function_map.update({f'tszs_{w}': _Function(function=wrap_non_picklable_objects(lambda x, w=w: ts_zscore(x, w)), name=f'tszs_{w}', arity=1) for w in tszs_win})

## arity >= 2
_extra_function_map.update({f'tszs_{w}_add': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_add(x, w)), name=f'tszs_{w}_add', arity=2) for w in tszs_win})
_extra_function_map.update({f'tszs_{w}_sub': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_sub(x, w)), name=f'tszs_{w}_sub', arity=2) for w in tszs_win})
_extra_function_map.update({f'tszs_{w}_mul': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_mul(x, w)), name=f'tszs_{w}_mul', arity=2) for w in tszs_win})
_extra_function_map.update({f'tszs_{w}_div': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_div(x, w)), name=f'tszs_{w}_div', arity=2) for w in tszs_win})
_extra_function_map.update({f'tszs_{w}_max': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_max(x, w)), name=f'tszs_{w}_max', arity=2) for w in tszs_win})
_extra_function_map.update({f'tszs_{w}_min': _Function(function=wrap_non_picklable_objects(lambda x, w=w: tszs_min(x, w)), name=f'tszs_{w}_min', arity=2) for w in tszs_win})
