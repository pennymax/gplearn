
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
            .rank(axis=1, method='first') # method first means assign different ranks on identical values
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

def compute_quantile_rets_fast(y, y_pred, w, quantiles):
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if np.all(np.isnan(y_pred)):
        return None, None
    
    rets = pd.DataFrame(y)
    factor = pd.DataFrame(y_pred)

    ## use y (return) to mask y_pred to set 0 on all invaid cells to nan
    factor = factor.mask(rets.isna())
    
    ret_qtl = []
    for _, grp in factor.stack().dropna().groupby(level=0, group_keys=True):
        ranks = grp.rank(method='first')   # method first means assign different ranks on identical values
        bins = np.linspace(ranks.min(), ranks.max(), quantiles+1)
        qtl = np.searchsorted(bins, ranks, side='left')
        qtl[qtl==0] = 1
        ret_qtl.append(pd.Series(qtl, index=ranks.index))
    if not ret_qtl:
        return None, None
    factor_quantiles = pd.concat(ret_qtl)

    stacked_rets = rets.stack()
    stacked_factor_quantiles = factor_quantiles
    grouped_returns = (
        stacked_rets
        .groupby([stacked_rets.index.get_level_values(0), stacked_factor_quantiles])
        .mean()
        .unstack()
        .mean()
        * _annulization
        ) 
    return grouped_returns, factor_quantiles.unstack()

def _quantile35_max(y, y_pred, w):
    if is_bad_data(y_pred):
        return 0
    
    res, _ = compute_quantile_rets_fast(y, y_pred, w, 35)
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
    
    res, _ = compute_quantile_rets_fast(y, y_pred, w, 35)
    if res is None:
        return 0
    else:
        return measure_monotonicity(res.values)

def _quantile35_longshort(y, y_pred, w):
    if is_bad_data(y_pred):
        return 0
    
    quantiles = 35
    res, _ = compute_quantile_rets_fast(y, y_pred, w, quantiles)
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

def _quantile35_longshort_fee_fast(y, y_pred, w):
    if is_bad_data(y_pred):
        return 0
    
    quantiles = 35
    res, factor_quantiles = compute_quantile_rets_fast(y, y_pred, w, quantiles)
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


###################################

def quantile_returns_and_groups(y, y_pred, quantile):
    fwdrets = pd.DataFrame(y)
    factor = pd.DataFrame(y_pred)

    ## use y (return) to mask y_pred to set 0 on all invaid cells to nan
    factor = factor.mask(fwdrets.isna())
    
    ret_qtl = []
    for _, grp in factor.stack().dropna().groupby(level=0, group_keys=True):
        ranks = grp.rank(method='first')   # method first means assign different ranks on identical values
        bins = np.linspace(ranks.min(), ranks.max(), quantile + 1)
        qtl = np.searchsorted(bins, ranks, side='left')
        qtl[qtl==0] = 1
        ret_qtl.append(pd.Series(qtl, index=ranks.index))
    if not ret_qtl:
        return None, None
    ## time x group_of_symbol (symbols number)
    stacked_factor_quantiles = pd.concat(ret_qtl)

    stacked_rets = fwdrets.stack()
    ## time x group_mean_return (group number)
    grouped_returns = (
        stacked_rets
        .groupby([stacked_rets.index.get_level_values(0), stacked_factor_quantiles])
        .mean()
        .unstack()
        )
    return grouped_returns, stacked_factor_quantiles.unstack()

def turnover_rates(df):
    df = df.notna()
    per_bar_changes = df.diff().abs().sum(axis=1) / 2
    per_bar_count = df.sum(axis=1).fillna(1)
    rates = per_bar_changes / per_bar_count.shift()
    rates = rates.replace([np.inf, -np.inf], 0)
    return rates

def calc_longshort_fee(factor_quantiles, quantile, fee_rate):
    # long
    pfl = factor_quantiles[factor_quantiles == quantile]
    long_rates = turnover_rates(pfl)
    # short
    pfl = factor_quantiles[factor_quantiles == 1]
    short_rates = turnover_rates(pfl)
    longshort_rates = (long_rates + short_rates) / 2
    longshort_fee = longshort_rates * fee_rate
    return longshort_fee

def quantile_longshort_returns(y, y_pred, w, quantile, fee_rate) -> pd.Series:
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if is_bad_data(y_pred):
        return pd.Series()
    
    grouped_returns, factor_quantiles = quantile_returns_and_groups(y, y_pred, quantile)
    longshort_rets = (grouped_returns[quantile] - grouped_returns[1]) / 2 ## not abs here as we need to calc camp and sub fee

    if fee_rate and fee_rate > 0:
        longshort_fee = calc_longshort_fee(factor_quantiles, quantile, fee_rate)
        longshort_rets = longshort_rets - longshort_fee
        # print(np.where(np.isinf(longshort_rets)))
    
    return longshort_rets

def cagr(returns, comp, annual_bars):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return -1000
    if comp:
        total_ret = (returns + 1).prod()
        cagr = total_ret ** (annual_bars / len(returns)) - 1
    else:
        cagr = returns.mean() * annual_bars
    return cagr

def total_return(returns, comp):
    if len(returns) < 10 and np.all(np.isnan(returns)):
        return -1000
    if comp:
        total_ret = returns.add(1).prod() - 1
    else:
        total_ret = returns.sum()
    return total_ret

def sharpe_simple(returns, annual_bars):
    if len(returns) < 10 and np.all(np.isnan(returns)):
        return -1000
    sharpe = np.sqrt(annual_bars) * returns.mean() / returns.std()
    return sharpe

def sharpe_fine(returns, comp, annual_bars):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return -1000
    cager_v = cagr(returns, comp, annual_bars)
    log_ret = np.log(returns + 1)
    log_ret = log_ret - log_ret.shift(1)
    annul_log_ret_vol = np.sqrt(annual_bars) * log_ret.std()
    sharpe = cager_v / annul_log_ret_vol
    return sharpe

# default using simple sharpe
def rolling_sharpe_sharpe(returns, window, annual_bars):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return -1000
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(annual_bars)
    rolling_sharpe = rolling_sharpe.dropna()
    if len(rolling_sharpe) < 100:
        return 0
    rolling_sharpe_sharpe = rolling_sharpe.mean() / rolling_sharpe.std()
    return rolling_sharpe_sharpe
        

##################################
##### Finess wrapper
##################################

annual_bar_8h = 365 * 3
fee_rate = 0.001

def fitness_quantile35_longshort_total_return_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=fee_rate)
    return total_return(longshort_rets, comp=True)

def fitness_quantile35_longshort_total_return_cumsum_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=fee_rate)
    return total_return(longshort_rets, comp=False)

def fitness_quantile35_longshort_cagr_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=fee_rate)
    return cagr(longshort_rets, comp=True, annual_bars=annual_bar_8h)

def fitness_quantile35_longshort_cagr_cumsum_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=fee_rate)
    return cagr(longshort_rets, comp=False, annual_bars=annual_bar_8h)

def fitness_quantile35_longshort_cagr_cumprod(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=0)
    return cagr(longshort_rets, comp=True, annual_bars=annual_bar_8h)

def fitness_quantile35_longshort_cagr_cumsum(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=0)
    return cagr(longshort_rets, comp=False, annual_bars=annual_bar_8h)

def fitness_quantile35_longshort_sharpe_fine_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=fee_rate)
    return sharpe_fine(longshort_rets, comp=True, annual_bars=annual_bar_8h)

def fitness_quantile35_longshort_sharpe_fine_cumsum_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=fee_rate)
    return sharpe_fine(longshort_rets, comp=False, annual_bars=annual_bar_8h)

def fitness_quantile35_longshort_sharpe_simple_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=fee_rate)
    return sharpe_simple(longshort_rets, annual_bars=annual_bar_8h)

def fitness_quantile35_longshort_rolling_sharpe_sharpe_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=fee_rate)
    # default simple sharpe and half year
    return rolling_sharpe_sharpe(longshort_rets, window=int(annual_bar_8h/2), annual_bars=annual_bar_8h)
    

_extra_fitness_map = {
    "rank_ic":                                      _Fitness(function=_rank_IC, greater_is_better=True),
    "rank_icir":                                    _Fitness(function=_rank_ICIR, greater_is_better=True),
    
    "quantile35_max":                               _Fitness(function=_quantile35_max, greater_is_better=True),
    "quantile35_mono":                              _Fitness(function=_quantile35_monotonicity, greater_is_better=True),
    
    ## Old CAGR
    "quantile35_longshort":                         _Fitness(function=_quantile35_longshort, greater_is_better=True), 
    "quantile35_longshort_fee":                     _Fitness(function=_quantile35_longshort_fee, greater_is_better=True), 
    "quantile35_longshort_fee_fast":                _Fitness(function=_quantile35_longshort_fee_fast, greater_is_better=True), 

    ## Total return
    "quantile35_longshort_total_return_cumprod_with_fee": _Fitness(function=fitness_quantile35_longshort_total_return_cumprod_with_fee, greater_is_better=True), 
    "quantile35_longshort_total_return_cumsum_with_fee": _Fitness(function=fitness_quantile35_longshort_total_return_cumsum_with_fee, greater_is_better=True), 
    
    ## CAGR
    'quantile35_longshort_cagr_cumprod_with_fee':   _Fitness(function=fitness_quantile35_longshort_cagr_cumprod_with_fee, greater_is_better=True), 
    'quantile35_longshort_cagr_cumsum_with_fee':    _Fitness(function=fitness_quantile35_longshort_cagr_cumsum_with_fee, greater_is_better=True), 
    'quantile35_longshort_cagr_cumprod':            _Fitness(function=fitness_quantile35_longshort_cagr_cumprod, greater_is_better=True), 
    'quantile35_longshort_cagr_cumsum':             _Fitness(function=fitness_quantile35_longshort_cagr_cumsum, greater_is_better=True), 

    ## Sharpe fine
    'quantile35_longshort_sharpe_fine_cumprod_with_fee':    _Fitness(function=fitness_quantile35_longshort_sharpe_fine_cumprod_with_fee, greater_is_better=True), 
    'quantile35_longshort_sharpe_fine_cumsum_with_fee':     _Fitness(function=fitness_quantile35_longshort_sharpe_fine_cumsum_with_fee, greater_is_better=True), 

    ## Sharpe simple
    'quantile35_longshort_sharpe_simple_cumprod_with_fee':    _Fitness(function=fitness_quantile35_longshort_sharpe_simple_cumprod_with_fee, greater_is_better=True), 

    ## Rolling sharpe sharpe
    'quantile35_longshort_rolling_sharpe_sharpe_with_fee':  _Fitness(function=fitness_quantile35_longshort_rolling_sharpe_sharpe_with_fee, greater_is_better=True), 
    
}



