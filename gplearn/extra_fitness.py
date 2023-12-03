
import numpy as np
import pandas as pd
from .fitness import _Fitness



def compute_IC(y, y_pred, w, rank_ic=True):
    y = y[w.astype(bool)]
    y_pred = y_pred[w.astype(bool)]
    if rank_ic:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y),axis = 1, method = "spearman").fillna(0)  # use fillna(0) to avoid getting high corr value if majorities are NaN
    else:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y),axis = 1, method = "pearson").fillna(0)
    return ic 

def _rank_IC(y, y_pred, w):
    ic = compute_IC(y, y_pred, w).mean()
    if np.isnan(ic):
        return 0
    else:
        return abs(ic)

def _rank_ICIR(y, y_pred, w):
    ics = compute_IC(y, y_pred, w)
    ic = ics.mean()
    ic_std = ics.std()
    icir = ic / ic_std
    if np.isnan(icir):
        return 0
    else:
        return abs(icir)
    
def compute_quantile_rets_ori(y, y_pred, w, quantiles):
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if np.all(np.isnan(y_pred)):
        return None
    
    annulization = 365 * 3 #默认8h  #TODO
    groups = np.array(range(quantiles)) + 1
    
    try:
        factor_quantiles = pd.DataFrame(y_pred).rank(axis=1,method='first').dropna(axis=0, how='all').apply(pd.qcut, q=quantiles, labels=groups, axis=1, duplicates='drop')
    except:
        return None

    rets = pd.DataFrame(y)
    return_series = {}
    for group in groups:
        returns_group = rets[factor_quantiles == group]
        return_series[group] = (returns_group.sum(axis=1) / returns_group.count(axis=1)).mean() * annulization # scale holding to 1 ; equal weights
    return return_series

def compute_quantile_rets(y, y_pred, w, quantiles):
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if np.all(np.isnan(y_pred)):
        return None
    
    rets = pd.DataFrame(y)
    factor = pd.DataFrame(y_pred)

    ## use y (return) to mask y_pred to set 0 on all invaid cells to nan
    factor = factor.mask(rets.isna())
    
    annulization = 365 * 3 #默认8h  #TODO
    groups = np.array(range(quantiles)) + 1
    
    try:
        factor_quantiles = (
            factor
            .rank(axis=1,method='first')
            .dropna(axis=0, how='all')
            .apply(pd.qcut, q=quantiles, labels=groups, axis=1, duplicates='drop')
            )
    except:
        return None
    
    stacked_rets = rets.stack()
    stacked_factor_quantiles = factor_quantiles.stack()
    grouped_returns = (
        stacked_rets
        .groupby([stacked_rets.index.get_level_values(0), stacked_factor_quantiles])
        .mean()
        .unstack()
        .mean()
        * annulization
        ) 
    return grouped_returns

def _quantile10_max(y, y_pred, w):
    res = compute_quantile_rets(y, y_pred, w, 10)
    if res is None:
        return 0
    else:
        return max(res.values)
    
def measure_monotonicity(data):
    ranks = [sorted(data).index(x) + 1 for x in data]
    rank_differences = [ranks[i] - ranks[i-1] for i in range(1, len(ranks))]
    positive_differences = sum(1 for diff in rank_differences if diff > 0)
    negative_differences = sum(1 for diff in rank_differences if diff < 0)
    monotonicity_score = abs(positive_differences - negative_differences) / len(data)
    return monotonicity_score

def _quantile10_monotonicity(y, y_pred, w):
    res = compute_quantile_rets(y, y_pred, w, 10)
    if res is None:
        return 0
    else:
        return measure_monotonicity(res.values)

def _quantile20_longshort(y, y_pred, w):
    quantiles = 20
    res = compute_quantile_rets(y, y_pred, w, quantiles)
    if res is None:
        return 0
    else:
        ret = abs(res[quantiles] - res[1]) / 2 # annualized longshort (not full time span）
        if np.isnan(ret):
            return 0
        else:
            return ret
    
weighted_rank_ic = _Fitness(function=_rank_IC,greater_is_better=True)
weighted_rank_icir = _Fitness(function=_rank_ICIR,greater_is_better=True)
weighted_quantile_max = _Fitness(function=_quantile10_max,greater_is_better=True)
weighted_quantile_mono = _Fitness(function=_quantile10_monotonicity,greater_is_better=True)
weighted_quantile_longshort = _Fitness(function=_quantile20_longshort,greater_is_better=True)

_extra_fitness_map = {
    "rank_ic": weighted_rank_ic,
    "rank_icir": weighted_rank_icir,
    "quantile_max": weighted_quantile_max,
    "quantile_mono": weighted_quantile_mono,
    "quantile_longshort": weighted_quantile_longshort, 
}



