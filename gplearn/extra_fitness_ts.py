
import numpy as np
import numbagg as nbgg
import polars as pl
import polars_ols as pls

from .fitness import _Fitness
from .extra_functions_ts import default_ts_zscore


_bad_fitness_val = -1000
_annual_bar_1h = 365 * 24
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

def convert_multiple_factor_values_to_OLS_predictions(
        dffct_vals: pl.DataFrame,
        if_zscore_pred: bool = True,
    ) -> pl.DataFrame:

    exclude_cols = ['y', 'orig_y']
    dfpred = (
        dffct_vals
        .select(
            [pl.col('y').least_squares.ols(col, add_intercept=True).alias(f'pred_{col}') for col in dffct_vals.columns if col not in exclude_cols]
        )
    )
    if if_zscore_pred:
        fct_columns = [col for col in dfpred.columns if col not in exclude_cols]
        rtns = dfpred.drop(exclude_cols).to_numpy()
        rtns_zs = default_ts_zscore(rtns)
        dfpred = (
            pl.DataFrame(rtns_zs, schema=fct_columns)
            .with_columns(
                y = dffct_vals['y'],
                orig_y = dffct_vals['orig_y'],
            )
            .fill_nan(0)
            .fill_null(0)
        )
    return dfpred
    
def convert_multiple_OLS_predictions_to_returns(
        dfpred: pl.DataFrame,
        fee_rate
    ) -> pl.DataFrame:
    
    exclude_cols = ['y', 'orig_y']
    dfrtn = (
        dfpred
        # .with_columns(pl.all().rolling_mean(5).clip(-2, 2))
        .with_columns(pl.exclude(exclude_cols).clip(-2, 2))
        .with_columns([
            pl.col(col).mul(pl.col('orig_y')) - pl.col(col).diff().abs().mul(fee_rate)
            for col in dfpred.columns if col not in exclude_cols
        ])
        .fill_nan(0)
        .fill_null(0)
    )
    return dfrtn

def convert_factor_value_to_returns(y, y_pred, fee_rate):
    ## check shape consistency
    if y.shape[0] != y_pred.shape[0]:
        return np.array([])
    
    ## check data quality
    if is_bad_data(y, y_pred):
        return np.array([])
        
    ## 
    dffct_vals = (
        pl.DataFrame(y_pred, orient='row', nan_to_null=True)
        .with_columns(
            y = pl.Series(default_ts_zscore(y)[:, 0]),
            orig_y = pl.Series(y[:, 0])
        )
        .fill_nan(0)
        .fill_null(0)
    )
    dfpred = convert_multiple_factor_values_to_OLS_predictions(dffct_vals)
    dfrtn = convert_multiple_OLS_predictions_to_returns(dfpred, fee_rate)
    rtn = dfrtn.drop('y', 'orig_y').to_numpy()

    return rtn
    
    
##################################
##### Finess wrapper
##################################

def ts_OLS_rtn_sharpe_simple_with_fee(y, y_pred, w):
    ls_ret = convert_factor_value_to_returns(y, y_pred, _fee_rate)
    return sharpe_simple(ls_ret, annual_periods=_annual_bar_1h)
    

_extra_fitness_map = {

    ## Sharpe simple
    'ts_OLS_rtn_sharpe_simple_with_fee':  _Fitness(function=ts_OLS_rtn_sharpe_simple_with_fee, greater_is_better=True), 
}


