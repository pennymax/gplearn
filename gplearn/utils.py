"""Utilities that are required by gplearn.

Most of these functions are slightly modified versions of some key utility
functions from scikit-learn that gplearn depends upon. They reside here in
order to maintain compatibility across different versions of scikit-learn.

"""

import numbers
import re
import pandas as pd

import numpy as np
from joblib import cpu_count

from _program import _Program
from extra_functions_2 import *


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _syntax_adapter(formulation:str):
    '''

    Args:
        formulation: 待解析的语法字符串

    Returns: 字典{'TOT':["condition a","condition b"]}

    '''
    elements = formulation.split(" ")
    iter_combination = ""
    adapted_dictionary={}
    register_flag = ""
    for pos,element in enumerate(elements):
        if element == "TOT" or element == "TRA" or element == "OOB":
            if len(iter_combination)>0:
                adapted_dictionary[register_flag].append(iter_combination)
            register_flag = element
            iter_combination = ""
            if element not in adapted_dictionary.keys():
                adapted_dictionary[element] = []
            continue
        iter_combination = iter_combination+element
        if pos==len(elements)-1:
            adapted_dictionary[register_flag].append(iter_combination)

    return adapted_dictionary


def check_floats(input_list,data):
    for item in input_list:
        if isinstance(item, float) and round(item, 3) == data:
            return True
    return False


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


def combine_factors__avg(exps, X, function_set, feature_names, use_rank=False, zscore=True):
    all_y_preds = []
    for exp in exps:
        y_pred = get_y_pred(exp, X, function_set, feature_names, debug=False)
        y_pred = cs_zscore(y_pred) if zscore else y_pred
        if use_rank:
            all_y_preds.append(pd.DataFrame(y_pred).rank(axis=1))
        else:
            all_y_preds.append(pd.DataFrame(y_pred))
    y_pred_mean = pd.concat(all_y_preds).groupby(level=0).mean().to_numpy()
    return y_pred_mean




