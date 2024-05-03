
import numpy as np
import pandas as pd
from .fitness import _Fitness


_bad_fitness_val = -1000
_annual_bar_8h = 365 * 3
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


   
def quantile_returns_and_groups(y, y_pred, quantile, select_quantiles_for_grouped_returns=None):
    mask = np.isnan(y)
    y_pred_masked = np.where(mask, np.nan, y_pred)

    ## 初始化分位数数组
    quantiles = np.full(y_pred_masked.shape, np.nan)

    ## 计算每个资产的排名和分位数，与 Pandas 版本相似
    ## we have to loop each line (cross section) as nan mask is different. No solution for non-loop vectorized yet...
    for i in range(y_pred_masked.shape[0]):
        cs = y_pred_masked[i, :]
        valid_mask = ~np.isnan(cs)
        if np.all(np.isnan(cs)):
            continue
        ranks = np.argsort(np.argsort(cs[valid_mask])) + 1
        bins = np.linspace(1, ranks.max(), quantile + 1)
        quantiles_values = np.searchsorted(bins, ranks, side='left')
        quantiles[i, valid_mask] = np.maximum(quantiles_values, 1)
    if np.all(np.isnan(quantiles)):
        return None, None

    ## 计算分组平均回报
    ## - select_quantiles_for_grouped_returns is used to save time, say if we only need [1, quantile] (save 60% time)
    if select_quantiles_for_grouped_returns:
        grouped_returns = np.array([np.nanmean(np.where(quantiles == q, y, np.nan), axis=1) 
                                    for q in select_quantiles_for_grouped_returns]).T
    ## - we can't use the same above code for all quantiles as it's slower 2x than below implementation
    else:
        dfy = pd.DataFrame(y)
        stacked_rets = dfy.stack()
        ## time x group_mean_return (group number)
        grouped_returns = (
            stacked_rets
            .groupby([stacked_rets.index.get_level_values(0), pd.DataFrame(quantiles).stack()])
            .mean()
            .unstack()
            .reindex(dfy.index)  ## to keep exact same shape with y
            .to_numpy()
            )

    assert(grouped_returns.shape[0] == quantiles.shape[0])
    return grouped_returns, quantiles

def turnover_rates(arr):
    # 计算每一行的变化（True 到 False 或 False 到 True）
    changes = np.diff(arr, axis=0)
    # print(valid_data.shape, changes.shape)
    per_bar_changes = np.sum(np.abs(changes), axis=1) / 2
    # 在 per_bar_changes 的开头添加一个 0 或 np.nan 以匹配长度
    per_bar_changes = np.insert(per_bar_changes, 0, 0)
    # 计算每一行的非 NaN 计数
    per_bar_count = np.sum(arr, axis=1).astype(float)
    # 防止除以零
    per_bar_count[per_bar_count == 0] = np.nan
    # 计算转换率
    rates = per_bar_changes / np.roll(per_bar_count, 1)
    # 处理无穷值
    rates[np.isinf(rates)] = 0
    return rates

def calc_longshort_fee(factor_quantiles, quantile, fee_rate):
    # long
    long_positions = factor_quantiles == quantile
    long_rates = turnover_rates(long_positions)
    # short
    short_positions = factor_quantiles == 1
    short_rates = turnover_rates(short_positions)
    # long short avg
    longshort_rates = (long_rates + short_rates) / 2
    # convert to fee
    longshort_fee = longshort_rates * fee_rate
    return longshort_fee

def quantile_longshort_returns(y, y_pred, w, quantile, fee_rate) -> pd.Series:
    if y.shape[0] != y_pred.shape[0]:
        return pd.Series()
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if is_bad_data(y, y_pred):
        return pd.Series()
    
    grouped_returns, factor_quantiles = quantile_returns_and_groups(y, y_pred, quantile)
    if grouped_returns is None or factor_quantiles is None or \
        np.all(np.isnan(grouped_returns)) or np.all(np.isnan(factor_quantiles)) or \
        grouped_returns.shape[1] != quantile:
        return pd.Series()
    longshort_rets = (grouped_returns[:, quantile-1] - grouped_returns[:, 0]) / 2 ## not abs here as we need to calc camp and sub fee
    # print('grouped_returns:', grouped_returns.shape)
    # print('factor_quantiles:', factor_quantiles.shape)

    if fee_rate and fee_rate > 0:
        longshort_fee = calc_longshort_fee(factor_quantiles, quantile, fee_rate)
        longshort_rets = longshort_rets - longshort_fee
        # print(np.where(np.isinf(longshort_rets)))
    
    return pd.Series(longshort_rets).fillna(0) ## treat nan as zero returns; keep original shape meantime


def sharpe_simple(returns, annual_bars):
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return _bad_fitness_val
    sharpe = np.sqrt(annual_bars) * returns.mean() / returns.std()
    if np.isnan(sharpe) or np.isinf(sharpe):
        return _bad_fitness_val
    return sharpe



##################################
##### Finess wrapper
##################################

## simple sharpe
def fitness_quantile35_longshort_sharpe_simple_cumprod_with_fee(y, y_pred, w):
    longshort_rets = quantile_longshort_returns(y, y_pred, w, quantile=35, fee_rate=_fee_rate)
    return sharpe_simple(longshort_rets, annual_bars=_annual_bar_8h)
    

_extra_fitness_map = {

    ## Sharpe simple
    'quantile35_longshort_sharpe_simple_cumprod_with_fee':  _Fitness(function=fitness_quantile35_longshort_sharpe_simple_cumprod_with_fee, greater_is_better=True), 

}


