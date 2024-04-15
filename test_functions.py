import time
import gplearn.extra_functions_3 as funset3
import gplearn.extra_functions_2 as funset2
import numpy as np
import timeit


def get_random_data(shape, nan_rate):
    data = np.random.randn(*shape)
    total_elements = data.size
    nan_elements_count = int(total_elements * nan_rate)
    indices = np.random.choice(total_elements, nan_elements_count, replace=False)
    data.flat[indices] = np.nan
    return data


def test_one_arity_func(func_name, number=100, **kargs):
    data = get_random_data((4000, 300), 0.2)
    # data = np.array([[1,2,3], [1,2,3], [1,2,3], [1,np.nan,3], [1,2,3]])
    
    r2 = eval(f'funset2.{func_name}')(data, **kargs)
    r3 = eval(f'funset3.{func_name}')(data, **kargs)
    eq = np.allclose(r2, r3, equal_nan=True)
    if not eq:
        print(r2)
        print('--------------')
        print(r3)

    r2_t = timeit.timeit(
        lambda: eval(f'funset2.{func_name}')(data, **kargs), setup='import gplearn.extra_functions_2 as funset2', number=number)
    r3_t = timeit.timeit(
        lambda: eval(f'funset3.{func_name}')(data, **kargs), setup='import gplearn.extra_functions_3 as funset3', number=number)
    r2_t = r2_t / number * 1000
    r3_t = r3_t / number * 1000
    
    suc_str = 'Pass' if eq else '!!Failed!!'
    print(f'[{suc_str}] [{func_name}] r2/r3 ratio: {r2_t/r3_t:.2f} ({r2_t:.1f}ms|{r3_t:.1f}ms)')


if __name__ == '__main__':

    test_one_arity_func('ta_LINEARREG', number=100, w=10)
    # test_one_arity_func('cs_rank', number=100)