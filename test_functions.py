import time
import gplearn.extra_functions_3 as funset3
import gplearn.extra_functions_2 as funset2
import numpy as np
import timeit
import os
from pprint import pformat

# os.environ["POLARS_MAX_THREADS"] = '1'
import polars as pl
print('Polars thread pool size:', pl.thread_pool_size())

from numpy.lib.stride_tricks import as_strided

from gplearn.extra_functions_3 import _extra_function_map


def get_random_data(shape, nan_rate):
    data = np.random.randn(*shape)
    total_elements = data.size
    nan_elements_count = int(total_elements * nan_rate)
    indices = np.random.choice(total_elements, nan_elements_count, replace=False)
    data.flat[indices] = np.nan
    return data


def test_compare_v2_v3_func(func_name, number=100, **kargs):
    print(f'compare function: {func_name}')
    data = get_random_data((4000, 300), 0.2)
    # data = np.array([[1,2,3], [1,2,3], [0,1.5,2], [1,np.nan,3], [4,5,6]])
    
    r2 = eval(f'funset2.{func_name}')(data, **kargs)
    r3 = eval(f'funset3.{func_name}')(data, **kargs)
    eq = np.allclose(r2, r3, equal_nan=True, rtol=0.9999)
    if not eq:
        print(r2)
        print('--------------')
        print(r3)
    # exit()

    r2_t = timeit.timeit(
        lambda: eval(f'funset2.{func_name}')(data, **kargs), setup='import gplearn.extra_functions_2 as funset2', number=number)
    r3_t = timeit.timeit(
        lambda: eval(f'funset3.{func_name}')(data, **kargs), setup='import gplearn.extra_functions_3 as funset3', number=number)
    r2_t = r2_t / number * 1000
    r3_t = r3_t / number * 1000
    
    suc_str = 'Pass' if eq else '!!Failed!!'
    print(f'[{suc_str}] [{func_name}] r2/r3 ratio: {r2_t/r3_t:.2f} ({r2_t:.1f}ms|{r3_t:.1f}ms)')


def test_v3_functions():
    data = get_random_data((4000, 300), 0.2)
    data2 = get_random_data((4000, 300), 0.2)
    data3 = get_random_data((4000, 300), 0.2)
    data4 = get_random_data((4000, 300), 0.2)
    failed_startpoint = []
    failed_lookahead = []
    failed_highnan = []
    failed_simulive = []

    print(f'== start testing {len(_extra_function_map)} functions ==')
    cnt = 0
    for k, v in _extra_function_map.items():
        cnt += 1
        print(f'\r[{cnt}|{len(_extra_function_map)}] test (arity=3) {k:<40}', end='')

        idx = data.shape[0]//2

        ## if depends on start point
        if v.arity == 1:
            d1 = v(data)
            d2 = v(data[idx:, :])
        elif v.arity == 2:
            d1 = v(data, data2)
            d2 = v(data[idx:, :], data2[idx:, :])
        elif v.arity == 3:
            d1 = v(data, data2, data3)
            d2 = v(data[idx:, :], data2[idx:, :], data3[idx:, :])
        elif v.arity == 4:
            d1 = v(data, data2, data3, data4)
            d2 = v(data[idx:, :], data2[idx:, :], data3[idx:, :], data4[idx:, :])
        if not np.allclose(d1[-10:, :], d2[-10:, :], equal_nan=True, rtol=0.9999):
            failed_startpoint.append(k)
        
        ## if look ahead data
        if v.arity == 1:
            d3 = v(data[:idx, :])
        elif v.arity == 2:
            d3 = v(data[:idx, :], data2[:idx, :])
        elif v.arity == 3:
            d3 = v(data[:idx, :], data2[:idx, :], data3[:idx, :])
        elif v.arity == 4:
            d3 = v(data[:idx, :], data2[:idx, :], data3[:idx, :], data4[:idx, :])
        if not np.allclose(d1[-idx-100:idx, :], d3[-100:, :], equal_nan=True, rtol=0.9999):
            failed_lookahead.append(k)
        
        ## check nan ratio
        nan_rt = np.sum(np.isnan(d1)) / d1.size
        if nan_rt > 0.3:
            failed_highnan.append((k, np.round(nan_rt, 3)))
        
        ## simulate live delta update
        if v.arity == 1:
            d_prebar = v(data[:-1, :])
            d_newbar_limit = v(data[-500:, :])
        elif v.arity == 2:
            d_prebar = v(data[:-1, :], data2[:-1, :])
            d_newbar_limit = v(data[-500:, :], data2[-500:, :])
        elif v.arity == 3:
            d_prebar = v(data[:-1, :], data2[:-1, :], data3[:-1, :])
            d_newbar_limit = v(data[-500:, :], data2[-500:, :], data3[-500:, :])
        elif v.arity == 4:
            d_prebar = v(data[:-1, :], data2[:-1, :], data3[:-1, :], data4[:-1, :])
            d_newbar_limit = v(data[-500:, :], data2[-500:, :], data3[-500:, :], data4[-500:, :])
        if not np.allclose(d_newbar_limit[-10:-1, :], d_prebar[-9:, :], equal_nan=True, rtol=0.9999):
            failed_simulive.append(k)
        
    print('\n---------------')
    print(f'[Test startpoint dependency] ({len(failed_startpoint)}) failed: {failed_startpoint}')
    print(f'[Test look ahead] failed ({len(failed_lookahead)}) : {failed_lookahead}')
    print(f'[Test high nan ratio] failed ({len(failed_highnan)}) : {failed_highnan}')
    print(f'[Test simulate live delta update] failed ({len(failed_simulive)}) : {failed_simulive}')


def test_func_perf(number=100):
    data = get_random_data((4000, 300), 0.2)
    data2 = get_random_data((4000, 300), 0.2)
    data3 = get_random_data((4000, 300), 0.2)
    data4 = get_random_data((4000, 300), 0.2)
    perf_data = []

    print(f'== start testing {len(_extra_function_map)} functions ==')
    cnt = 0
    for k, v in _extra_function_map.items():
        cnt += 1
        def _get_lambda_func(v):
            if v.arity == 1:
                return lambda: v(data)
            elif v.arity == 2:
                return lambda: v(data, data2)
            elif v.arity == 3:
                return lambda: v(data, data2, data3)
            elif v.arity == 4:
                return lambda: v(data, data2, data3, data4)

        p = timeit.timeit(
            _get_lambda_func(v), 
            setup='import gplearn.extra_functions_3 as funset3', 
            number=number
        )
        p = p / number * 1000
        perf_data.append([k, p])
        print(f'\r[{cnt}|{len(_extra_function_map)}] test (arity={v.arity}) {k:<20} {p:10.2f}ms', end='')

    print()
    print(pformat(sorted(perf_data, key=lambda x: x[1], reverse=True)[:10]))
    (
        pl.DataFrame(perf_data, schema={'func': pl.String, 'avg_t': pl.Float64})
        .with_columns(
            (pl.col('avg_t') / pl.col('avg_t').min()).cast(pl.Int64).alias('score')
        ).sort('avg_t', descending=True)
        .write_csv('./func_perf_data.csv')
    )
            

if __name__ == '__main__':

    # test_compare_v2_v3_func('ts_autocorr', number=100, w=60, l=1)
    # test_compare_v2_v3_func('ts_quantile', number=100, w=10, q=0.25)
    # test_compare_v2_v3_func('ts_skew', number=100, w=10)
    # test_compare_v2_v3_func('ta_HTTRENDLINE', number=100)
    test_compare_v2_v3_func('ta_APO', number=100, fastperiod=12, slowperiod=26, matype=0)

    # test_v3_functions()

    # test_func_perf(100)