
import numpy as np
import pandas as pd
from .fitness import _Fitness


_bad_fitness_val = -1000
_annual_bar_8h = 365 * 3
_fee_rate = 0.001

def is_bad_data(y, y_pred, max_full_nan_cs_rate=0.3, min_valid_rate=0.7):
    ## check if too many full nan cs
    total_cs = y_pred.shape[0]
    all_na_cs = np.sum(np.all(np.isnan(y_pred), axis=1))
    too_many_invalid_full_nan_cs = all_na_cs / total_cs > max_full_nan_cs_rate
    
    ## check if too 
    cs_nan_y_pred = np.sum(~np.isnan(y_pred), axis=1)
    cs_nan_y = np.sum(~np.isnan(y), axis=1)
    rate = cs_nan_y_pred / cs_nan_y
    rate = np.where(np.isinf(rate), 0, rate)
    too_low_mean_valid_rate = np.mean(rate) < min_valid_rate
    if too_many_invalid_full_nan_cs or too_low_mean_valid_rate:
        return True
    else:
        return False

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
    if ret_qtl:
        ## time x group_of_symbol (symbols number)
        stacked_factor_quantiles = pd.concat(ret_qtl)
    else:
        return pd.DataFrame(), pd.DataFrame()

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
    rates = per_bar_changes / per_bar_count.shift().replace(0, np.nan)
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
    if is_bad_data(y, y_pred):
        return pd.Series()
    
    grouped_returns, factor_quantiles = quantile_returns_and_groups(y, y_pred, quantile)
    if grouped_returns.empty or factor_quantiles.empty or quantile not in grouped_returns.columns or 1 not in grouped_returns.columns:
        return pd.Series()
    longshort_rets = (grouped_returns[quantile] - grouped_returns[1]) / 2 ## not abs here as we need to calc camp and sub fee

    if fee_rate and fee_rate > 0:
        longshort_fee = calc_longshort_fee(factor_quantiles, quantile, fee_rate)
        longshort_rets = longshort_rets - longshort_fee
        # print(np.where(np.isinf(longshort_rets)))
    
    return longshort_rets

def cagr(returns, comp, annual_bars):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return _bad_fitness_val
    if comp:
        total_ret = (returns + 1).prod()
        cagr = total_ret ** (annual_bars / len(returns)) - 1
    else:
        cagr = returns.mean() * annual_bars
    return _bad_fitness_val if np.isnan(cagr) else cagr

def total_return(returns, comp):
    if returns.empty:
        return _bad_fitness_val
    if comp:
        total_ret = returns.add(1).prod() - 1
    else:
        total_ret = returns.sum()
    return total_ret

def sharpe_simple(returns, annual_bars):
    if len(returns) < 10 and np.all(np.isnan(returns)):
        return _bad_fitness_val
    sharpe = np.sqrt(annual_bars) * returns.mean() / returns.std()
    if np.isnan(sharpe) or np.isinf(sharpe):
        return _bad_fitness_val
    return sharpe

def sharpe_fine(returns, comp, annual_bars):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return _bad_fitness_val
    cager_v = cagr(returns, comp, annual_bars)
    log_ret = np.log(returns + 1)
    log_ret = log_ret - log_ret.shift(1)
    annul_log_ret_vol = np.sqrt(annual_bars) * log_ret.std()
    sharpe = cager_v / annul_log_ret_vol
    if np.isnan(sharpe) or np.isinf(sharpe):
        return _bad_fitness_val
    return sharpe

# default using simple sharpe
def rolling_sharpe_sharpe(returns, window, annual_bars):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return _bad_fitness_val
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(annual_bars)
    rolling_sharpe = rolling_sharpe.dropna()
    if len(rolling_sharpe) < 100:
        return 0
    rolling_sharpe_sharpe = rolling_sharpe.mean() / rolling_sharpe.std()
    if np.isnan(rolling_sharpe_sharpe) or np.isinf(rolling_sharpe_sharpe):
        return _bad_fitness_val
    return rolling_sharpe_sharpe

def monotonicity(y, y_pred, w, quantile):
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if is_bad_data(y, y_pred):
        return _bad_fitness_val
    
    grouped_returns, _ = quantile_returns_and_groups(y, y_pred, quantile)
    grouped_returns_mean = grouped_returns.mean()
    # print(grouped_returns_mean)
    
    if len(grouped_returns) < 1:
        return _bad_fitness_val
    ranks = [sorted(grouped_returns_mean).index(x) + 1 for x in grouped_returns_mean]
    rank_differences = [ranks[i] - ranks[i-1] for i in range(1, len(ranks))]
    positive_differences = sum(1 for diff in rank_differences if diff > 0)
    negative_differences = sum(1 for diff in rank_differences if diff < 0)
    monotonicity_score = abs(positive_differences - negative_differences) / len(grouped_returns_mean) if len(grouped_returns_mean) > 0 else _bad_fitness_val
    return monotonicity_score

def compute_IC(y, y_pred, w, rank_ic=True):
    if is_bad_data(y, y_pred):
        return 0
    y = y[w.astype(bool)]
    y_pred = y_pred[w.astype(bool)]
    if rank_ic:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y),axis = 1, method = "spearman").fillna(0)  # use fillna(0) to avoid getting high corr value if majorities are NaN
    else:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y),axis = 1, method = "pearson").fillna(0)
    return ic 

def ic(y, y_pred, w, rank_ic):
    if is_bad_data(y, y_pred):
        return _bad_fitness_val
    ic = compute_IC(y, y_pred, w, rank_ic).mean()
    if np.isnan(ic):
        return _bad_fitness_val
    else:
        return ic

def icir(y, y_pred, w, rank_ic):
    if is_bad_data(y, y_pred):
        return _bad_fitness_val
    ics = compute_IC(y, y_pred, w, rank_ic)
    ic = ics.mean()
    ic_std = ics.std()
    icir = ic / ic_std
    if np.isnan(icir):
        return _bad_fitness_val
    else:
        return icir

def quantile_avg_turover_rate(y, y_pred, w, quantile):
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if is_bad_data(y, y_pred):
        return _bad_fitness_val
    
    _, factor_quantiles = quantile_returns_and_groups(y, y_pred, quantile)
    if factor_quantiles.empty:
        return _bad_fitness_val

    # long
    pfl = factor_quantiles[factor_quantiles == quantile]
    long_rates = turnover_rates(pfl)
    # short
    pfl = factor_quantiles[factor_quantiles == 1]
    short_rates = turnover_rates(pfl)
    longshort_rates = (long_rates + short_rates) / 2
    avg_longshort_rates = longshort_rates.mean()
    return avg_longshort_rates

def quantile_longshort_dd_top_n_mean(returns, topn):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return _bad_fitness_val
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    if len(drawdown) < 10:
        return _bad_fitness_val
    return drawdown.nsmallest(topn).mean()


def quantile_longshort_sortino(returns, annual_bars, risk_free_rate=0):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return _bad_fitness_val
    excess_return = returns - risk_free_rate
    negative_return = returns[returns < risk_free_rate]
    downside_std = negative_return.std()
    if downside_std == 0:
        return _bad_fitness_val
    sortino_ratio = excess_return.mean() / downside_std
    return sortino_ratio * np.sqrt(annual_bars)

def win_rate(returns):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return _bad_fitness_val
    positive_returns = returns[returns > 0].count()
    total_trades = returns.count()
    win_rate = positive_returns / total_trades
    return win_rate


##################################
##### Finess wrapper
##################################

## total returns
def fitness_quantile35_longshort_total_return_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return total_return(longshort_rets, comp=True)

def fitness_quantile35_longshort_total_return_cumsum_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return total_return(longshort_rets, comp=False)

## cagr
def fitness_quantile35_longshort_cagr_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return cagr(longshort_rets, comp=True, annual_bars=_annual_bar_8h)

def fitness_quantile35_longshort_cagr_cumsum_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return cagr(longshort_rets, comp=False, annual_bars=_annual_bar_8h)

def fitness_quantile35_longshort_cagr_cumprod(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=0)
    return cagr(longshort_rets, comp=True, annual_bars=_annual_bar_8h)

def fitness_quantile35_longshort_cagr_cumsum(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=0)
    return cagr(longshort_rets, comp=False, annual_bars=_annual_bar_8h)

## sharpe
def fitness_quantile35_longshort_sharpe_fine_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return sharpe_fine(longshort_rets, comp=True, annual_bars=_annual_bar_8h)

def fitness_quantile35_longshort_sharpe_fine_cumsum_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return sharpe_fine(longshort_rets, comp=False, annual_bars=_annual_bar_8h)

## rolling sharpe sharpe
def fitness_quantile35_longshort_sharpe_simple_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return sharpe_simple(longshort_rets, annual_bars=_annual_bar_8h)

def fitness_quantile35_longshort_rolling_sharpe_sharpe_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    # default simple sharpe and half year
    return rolling_sharpe_sharpe(longshort_rets, window=int(_annual_bar_8h/2), annual_bars=_annual_bar_8h)

## monotonicity
def fitness_quantile35_monotonicity_with_fee(y, y_pred, w):
    return monotonicity(y, y_pred, w, quantile=35)

## IC & ICIR
def fitness_rank_ic(y, y_pred, w):
    return ic(y, y_pred, w, rank_ic=True)

def fitness_rank_icir(y, y_pred, w):
    return icir(y, y_pred, w, rank_ic=True)

## turnover
def fitness_quantile35_longshort_avg_turnover_rate(y, y_pred, w):
    return quantile_avg_turover_rate(y, y_pred, w, quantile=35)

## mdd
def fitness_quantile35_longshort_cumprod_mdd_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return quantile_longshort_dd_top_n_mean(longshort_rets, 1)

def fitness_quantile35_longshort_cumprod_top5_avg_mdd_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return quantile_longshort_dd_top_n_mean(longshort_rets, 5)

## sortino
def fitness_quantile35_longshort_sortino_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return quantile_longshort_sortino(longshort_rets, annual_bars=_annual_bar_8h)

## win rate
def fitness_quantile35_longshort_winrate_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return win_rate(longshort_rets)

## max avg long return
def fitness_quantile35_longshort_max_avg_long_return_with_fee(y, y_pred, w):
    ...
    

_extra_fitness_map = {

    # "quantile35_max":                               _Fitness(function=_quantile35_max, greater_is_better=True),

    ################################################

    ## IC & ICIR
    "rank_ic":                                              _Fitness(function=fitness_rank_ic, greater_is_better=True),
    "rank_icir":                                            _Fitness(function=fitness_rank_icir, greater_is_better=True),

    ## Total return
    "quantile35_longshort_total_return_cumprod_with_fee":   _Fitness(function=fitness_quantile35_longshort_total_return_cumprod_with_fee, greater_is_better=True), 
    "quantile35_longshort_total_return_cumsum_with_fee":    _Fitness(function=fitness_quantile35_longshort_total_return_cumsum_with_fee, greater_is_better=True), 
    
    ## CAGR
    'quantile35_longshort_cagr_cumprod_with_fee':           _Fitness(function=fitness_quantile35_longshort_cagr_cumprod_with_fee, greater_is_better=True), 
    'quantile35_longshort_cagr_cumsum_with_fee':            _Fitness(function=fitness_quantile35_longshort_cagr_cumsum_with_fee, greater_is_better=True), 
    'quantile35_longshort_cagr_cumprod':                    _Fitness(function=fitness_quantile35_longshort_cagr_cumprod, greater_is_better=True), 
    'quantile35_longshort_cagr_cumsum':                     _Fitness(function=fitness_quantile35_longshort_cagr_cumsum, greater_is_better=True), 

    ## Sharpe fine
    'quantile35_longshort_sharpe_fine_cumprod_with_fee':    _Fitness(function=fitness_quantile35_longshort_sharpe_fine_cumprod_with_fee, greater_is_better=True), 
    'quantile35_longshort_sharpe_fine_cumsum_with_fee':     _Fitness(function=fitness_quantile35_longshort_sharpe_fine_cumsum_with_fee, greater_is_better=True), 

    ## Sharpe simple
    'quantile35_longshort_sharpe_simple_cumprod_with_fee':  _Fitness(function=fitness_quantile35_longshort_sharpe_simple_cumprod_with_fee, greater_is_better=True), 

    ## Rolling sharpe sharpe
    'quantile35_longshort_rolling_sharpe_sharpe_with_fee':  _Fitness(function=fitness_quantile35_longshort_rolling_sharpe_sharpe_with_fee, greater_is_better=True), 

    ## Monotonicity
    'quantile35_monotonicity':                              _Fitness(function=fitness_quantile35_monotonicity_with_fee, greater_is_better=True),

    ## turnover rate
    'quantile35_longshort_avg_turnover_rate':               _Fitness(function=fitness_quantile35_longshort_avg_turnover_rate, greater_is_better=True),

    ## MDD
    'quantile35_longshort_cumprod_mdd_with_fee':            _Fitness(function=fitness_quantile35_longshort_cumprod_mdd_with_fee, greater_is_better=True),
    'quantile35_longshort_cumprod_top5_avg_mdd_with_fee':   _Fitness(function=fitness_quantile35_longshort_cumprod_top5_avg_mdd_with_fee, greater_is_better=True), 

    ## Sortino
    'quantile35_longshort_sortino_cumprod_with_fee':        _Fitness(function=fitness_quantile35_longshort_sortino_cumprod_with_fee, greater_is_better=True),

    ## win rate
    'quantile35_longshort_winrate_with_fee':                _Fitness(function=fitness_quantile35_longshort_winrate_with_fee, greater_is_better=True),

}


