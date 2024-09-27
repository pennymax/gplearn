
import numpy as np
import pandas as pd
import numbagg as nbgg
import bottleneck as bn

from .fitness import _Fitness


_bad_fitness_val = -1000
_annual_bar_8h = 365 * 3
_annual_bar_4h = 365 * 6
_fee_rate = 0.001


def is_bad_data(y, y_pred, max_full_nan_cs_rate=0.3, min_valid_rate=0.7):
    # 检查是否有过多的完全是 NaN 的行（cross-sections）
    all_na_cs_mask = np.all(np.isnan(y_pred), axis=1)
    too_many_invalid_full_nan_cs = np.mean(all_na_cs_mask) > max_full_nan_cs_rate
    
    # 计算 y_pred 和 y 中非 NaN 值的数量
    valid_y_pred = np.sum(~np.isnan(y_pred), axis=1)
    valid_y = np.sum(~np.isnan(y), axis=1)

    # 避免除以零的情况
    valid_mask = valid_y != 0
    valid_rates = valid_y_pred[valid_mask] / valid_y[valid_mask]

    # 计算平均有效率，并检查是否过低
    too_low_mean_valid_rate = np.mean(valid_rates) < min_valid_rate if valid_rates.size > 0 else True

    return too_many_invalid_full_nan_cs or too_low_mean_valid_rate


def sharpe_simple(x: np.ndarray, annual_periods: int) -> float:
    if len(x) < 10 or np.all(np.isnan(x)):
        return _bad_fitness_val
    mean = nbgg.nanmean(x)
    std = nbgg.nanstd(x, ddof=0)
    sr = np.sqrt(annual_periods) * mean / std
    if np.isnan(sr) or np.isinf(sr):
        return _bad_fitness_val
    return sr

def get_topmean_and_turnover(factor_val, y_masked, pfl_cnt):
    ## trick: bn.replace save several ms. comparing with: np.where(np.isnan(fct), np.inf, fct)
    ## but we need to handle -np.inf since -fct in call for long_rets
    ## note: bn.replace is inplace replacement!
    factor_val = bn.replace(factor_val, np.nan, np.inf)
    factor_val = bn.replace(factor_val, -np.inf, np.inf)

    ## divide array by kth index in sorted array (but not sorted for other parts)
    indices = bn.argpartition(factor_val, kth=pfl_cnt, axis=1)[:, :pfl_cnt]
    # print('indices', indices)

    ## get mean of top k elements
    top_rets = np.take_along_axis(y_masked, indices, axis=1)
    pfl_rets = bn.nanmean(top_rets, axis=1)

    ## get turn over by comparing symbol changes
    pfl = np.full_like(factor_val, False, dtype=bool)
    pfl[np.arange(factor_val.shape[0])[:, None], indices] = True
    pfl[np.isnan(factor_val)] = False
    pfl_diff = np.abs(np.diff(pfl, axis=0)) # result of diff from 2nd
    turnover_from2nd = bn.nansum(pfl_diff, axis=1) / 2 / pfl_cnt
    
    return pfl_rets, turnover_from2nd, pfl


def convert_factor_value_to_returns(y, y_pred, mask_time_and_universe, fee_rate, pfl_cnt):
    ## check shape consistency
    if y.shape[0] != y_pred.shape[0]:
        return np.array([])
    
    ## mask_time_and_universe contains offset months data and universe data.
    ## here we need to remove offset months data
    ## see mask_time_and_universe define in gplearn pipeline_steps.py
    mask_time = np.any(mask_time_and_universe, axis=1)  # flag rows w/o full nans
    mask = mask_time_and_universe[mask_time]            # filter out rows w/ full nans, a.k.a offset months data
    y = y[mask_time]                                    # filter out offset months data
    y_pred = y_pred[mask_time]                          # filter out offset months data
    # print(mask_time_and_universe.shape, mask.shape, y.shape, y_pred.shape)
    
    ## mask fct on universal symbols
    fct_masked = np.where(mask, y_pred, np.nan)
    
    ## mask y on valid fct
    y_masked = np.where(np.isnan(fct_masked), np.nan, y)
    
    ## check data quality
    if is_bad_data(y_masked, fct_masked):
         return np.array([])
    
    ## get long and short mean return on each time cross section
    short_rets, short_to, _ = get_topmean_and_turnover(fct_masked, y_masked, pfl_cnt)
    long_rets, long_to, _   = get_topmean_and_turnover(-fct_masked, y_masked, pfl_cnt)
    
    ## get longshort return with fee 
    longshort_ret = (long_rets - short_rets) * 0.5
    longshort_fee = (long_to + short_to) * fee_rate * 0.5
    longshort_ret[1:] -= longshort_fee
    longshort_ret[0] -= fee_rate * 0.5  # only approximate correct
    
    return longshort_ret
    
    
##################################
##### Finess wrapper
##################################

def pfl5_longshort_sharpe_simple_with_fee(y, y_pred, w):
    ls_ret = convert_factor_value_to_returns(y, y_pred, w, _fee_rate, pfl_cnt=5)
    return sharpe_simple(ls_ret, annual_periods=_annual_bar_8h)

def pfl5_longshort_sharpe_simple_with_fee_4h(y, y_pred, w):
    ls_ret = convert_factor_value_to_returns(y, y_pred, w, _fee_rate, pfl_cnt=5)
    return sharpe_simple(ls_ret, annual_periods=_annual_bar_4h)
    

_extra_fitness_map = {

    ## Sharpe simple (8h)
    'pfl5_longshort_sharpe_simple_with_fee':  _Fitness(function=pfl5_longshort_sharpe_simple_with_fee, greater_is_better=True), 

    ## Sharpe simple (4h)
    'pfl5_longshort_sharpe_simple_with_fee_4h':  _Fitness(function=pfl5_longshort_sharpe_simple_with_fee_4h, greater_is_better=True), 
}

