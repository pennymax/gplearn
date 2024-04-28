import re
import pandas as pd
import time

from gplearn._program import _Program
# from gplearn.extra_functions_3 import *
from gplearn.extra_functions_2 import *
from gplearn.utils import *

from contextlib import contextmanager

@contextmanager
def timer(label, if_print=True):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        if if_print:
            print(f"== {label}: {end - start:.4f} seconds ==")


def convert_expression_to_gp_program(expression: str, function_set: dict, feature_names: list) -> _Program:
    ## convert expression to list function obj and feature index
    tokens = re.findall(r'[\w.]+|\(|\)|,', expression)
    tokens = [t for t in tokens if t not in ['(', ')', ',']]
    # print(tokens)
    program = []
    for token in tokens:
        if token in function_set:
            program.append(function_set[token])
        elif token in feature_names:
            program.append(feature_names.index(token))
        elif token.isdigit():
            program.append(int(token))
        elif token.replace('.', '', 1).isdigit():
            program.append(float(token))
        else:
            print(f'  !! unknown token found! {token}')
            return None
    # print(program)

    ## get arity dict
    arities = {}
    for function in function_set.values():
        arity = function.arity
        arities[arity] = arities.get(arity, [])
        arities[arity].append(function)

    ## construct _Program obj
    params = {
            'function_set': function_set,
            'arities': arities,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'init_depth': (2, 6),
            ## must inputs
            'init_method': 'half and half',
            'const_range': (-1.0, 1.0),
            'metric': 'mean absolute error',
            'p_point_replace': 0.05,
            'parsimony_coefficient': 0.1,
            'random_state': check_random_state(None),
            }
    gp = _Program(program=program, **params)
    return gp


def get_y_pred(exp, X, function_set, feature_names, debug=True):
    gp = convert_expression_to_gp_program(exp, function_set, feature_names)
    if not gp:
        print(f'  !!!! invalid expression {exp} !!!!')
        return None
    if debug:
        print(f'_Program print: {gp}')
    y_pred = gp.execute_3D(X)
    return y_pred

def winsorize_2d(arr):
    Q1 = np.nanpercentile(arr, 25, axis=1, keepdims=True)  # 计算每行的Q1
    Q3 = np.nanpercentile(arr, 75, axis=1, keepdims=True)  # 计算每行的Q3
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    arr_winsorized = np.where(np.isnan(arr), arr, np.clip(arr, lower_bound, upper_bound))
    return arr_winsorized

def combine_factors__avg(exps, X, function_set, feature_names, use_rank=False, zscore=True, remove_outlier=False):
    all_y_preds = []
    for exp in exps:
        y_pred = get_y_pred(exp, X, function_set, feature_names, debug=False)
        if remove_outlier:
            y_pred = winsorize_2d(y_pred)
        y_pred = cs_zscore(y_pred) if zscore else y_pred
        if use_rank:
            all_y_preds.append(pd.DataFrame(y_pred).rank(axis=1))
        else:
            all_y_preds.append(pd.DataFrame(y_pred))
    y_pred_mean = pd.concat(all_y_preds).groupby(level=0).mean().to_numpy()
    return y_pred_mean
