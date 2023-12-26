
import numpy as np
import pandas as pd
from .fitness import _Fitness


_annulization = 365 * 3 #默认8h  #TODO
_fee = 0.001

def is_bad_data(y_pred, thresh=0.7):
    total_cs = y_pred.shape[0]
    all_na_cs = np.sum(np.all(np.isnan(y_pred), axis=1))
    return all_na_cs / total_cs >= thresh


def compute_IC(y, y_pred, w, rank_ic=True):
    if is_bad_data(y_pred):
        return 0
    y = y[w.astype(bool)]
    y_pred = y_pred[w.astype(bool)]
    if rank_ic:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y),axis = 1, method = "spearman").fillna(0)  # use fillna(0) to avoid getting high corr value if majorities are NaN
    else:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y),axis = 1, method = "pearson").fillna(0)
    return ic 

def _rank_IC(y, y_pred, w):
    if is_bad_data(y_pred):
        return 0
    ic = compute_IC(y, y_pred, w).mean()
    if np.isnan(ic):
        return 0
    else:
        return abs(ic)

def _rank_ICIR(y, y_pred, w):
    if is_bad_data(y_pred):
        return 0
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
    
    groups = np.array(range(quantiles)) + 1
    
    try:
        factor_quantiles = pd.DataFrame(y_pred).rank(axis=1,method='first').dropna(axis=0, how='all').apply(pd.qcut, q=quantiles, labels=groups, axis=1, duplicates='drop')
    except:
        return None

    rets = pd.DataFrame(y)
    return_series = {}
    for group in groups:
        returns_group = rets[factor_quantiles == group]
        return_series[group] = (returns_group.sum(axis=1) / returns_group.count(axis=1)).mean() * _annulization # scale holding to 1 ; equal weights
    return return_series

def turnover_rate(df):
    df = df.notna()
    per_bar_changes = df.diff().abs().sum(axis=1) / 2
    per_bar_count = df.sum(axis=1)
    rates = per_bar_changes / per_bar_count.shift()
    avg_rate = rates.mean()
    return avg_rate

def compute_quantile_rets(y, y_pred, w, quantiles):
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if np.all(np.isnan(y_pred)):
        return None, None
    
    rets = pd.DataFrame(y)
    factor = pd.DataFrame(y_pred)

    ## use y (return) to mask y_pred to set 0 on all invaid cells to nan
    factor = factor.mask(rets.isna())
    
    groups = np.array(range(quantiles)) + 1
    
    try:
        factor_quantiles = (
            factor
            .rank(axis=1,method='first')
            .dropna(axis=0, how='all')
            .apply(pd.qcut, q=quantiles, labels=groups, axis=1, duplicates='drop')
            )
    except:
        return None, None
    
    stacked_rets = rets.stack()
    stacked_factor_quantiles = factor_quantiles.stack()
    grouped_returns = (
        stacked_rets
        .groupby([stacked_rets.index.get_level_values(0), stacked_factor_quantiles])
        .mean()
        .unstack()
        .mean()
        * _annulization
        ) 
    return grouped_returns, factor_quantiles

def _quantile35_max(y, y_pred, w):
    if is_bad_data(y_pred):
        return 0
    
    res, _ = compute_quantile_rets(y, y_pred, w, 35)
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

def _quantile35_monotonicity(y, y_pred, w):
    if is_bad_data(y_pred):
        return 0
    
    res, _ = compute_quantile_rets(y, y_pred, w, 35)
    if res is None:
        return 0
    else:
        return measure_monotonicity(res.values)

def _quantile35_longshort(y, y_pred, w):
    if is_bad_data(y_pred):
        return 0
    
    quantiles = 35
    res, _ = compute_quantile_rets(y, y_pred, w, quantiles)
    if res is None:
        return 0
    else:
        ret = abs(res[quantiles] - res[1]) / 2 # annualized longshort (not full time span）
        if np.isnan(ret):
            return 0
        else:
            return ret

def _quantile35_longshort_fee(y, y_pred, w):
    if is_bad_data(y_pred):
        return 0
    
    quantiles = 35
    res, factor_quantiles = compute_quantile_rets(y, y_pred, w, quantiles)
    if res is None or factor_quantiles is None:
        return 0
    else:
        if quantiles not in res.index or 1 not in res.index:
            return 0
        ret = abs(res[quantiles] - res[1]) / 2 # annualized longshort (not full time span）
        if np.isnan(ret):
            return 0
        else:
            long_turnover = turnover_rate(factor_quantiles[factor_quantiles == quantiles])
            short_turnover = turnover_rate(factor_quantiles[factor_quantiles == 1])
            avg_turnover = (long_turnover + short_turnover) / 2
            ret_fee = ret - avg_turnover * _fee * _annulization
            return ret_fee
    
weighted_rank_ic = _Fitness(function=_rank_IC,greater_is_better=True)
weighted_rank_icir = _Fitness(function=_rank_ICIR,greater_is_better=True)
weighted_quantile35_max = _Fitness(function=_quantile35_max,greater_is_better=True)
weighted_quantile35_mono = _Fitness(function=_quantile35_monotonicity,greater_is_better=True)
weighted_quantile35_longshort = _Fitness(function=_quantile35_longshort,greater_is_better=True)
weighted_quantile35_longshort_fee = _Fitness(function=_quantile35_longshort_fee,greater_is_better=True)

_extra_fitness_map = {
    "rank_ic": weighted_rank_ic,
    "rank_icir": weighted_rank_icir,
    "quantile35_max": weighted_quantile35_max,
    "quantile35_mono": weighted_quantile35_mono,
    "quantile35_longshort": weighted_quantile35_longshort, 
    "quantile35_longshort_fee": weighted_quantile35_longshort_fee, 
}



