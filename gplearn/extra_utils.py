import re
import pandas as pd

from gplearn._program import _Program
from gplearn.extra_functions_2 import *
from gplearn.utils import *



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
