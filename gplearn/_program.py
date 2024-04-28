"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from copy import copy
import numpy as np
from sklearn.utils.random import sample_without_replacement
from .functions import _Function
from .utils import check_random_state
import warnings
from copy import  deepcopy
warnings.filterwarnings("ignore")
import time



class FactorValueCache:

    def __init__(self, max_cnt):
        self.ttl = 0
        self.ttl_hit = 0
        self.max_cnt = max_cnt
        self.cache = {}
    
    def get_sorted_cache(self, reverse=False):
        if self.cache:
            return sorted(self.cache.items(), key=lambda x: x[1]['cnt'], reverse=reverse)
        else:
            return {}
    
    def put(self, key, value):
        # print('put')
        if key in self.cache:
            return
        else:
            if len(self.cache) >= self.max_cnt:
                sorted_cache = self.get_sorted_cache()
                min_key = sorted_cache[0][0]
                if min_key:
                    del self.cache[min_key]
            self.cache[key] = {'cnt': 1, 'val': value}

    def get(self, key):
        self.ttl += 1
        if key in self.cache and np.size(self.cache[key]['val']) > 0:
            self.cache[key]['cnt'] += 1
            self.ttl_hit += 1
            return self.cache[key]['val'].astype(np.float64)
        else:
            return np.array([])


class FactorValueCacheWhitList(FactorValueCache):
    def __init__(self, whiteList_keys):
        super().__init__(0)
        for k in whiteList_keys:
            self.cache[k] = {'cnt': 0, 'val': np.array([])}
    
    def put(self, key, value: np.ndarray):
        # print('put')
        if key in self.cache and self.cache[key]['cnt'] < 1:
            self.cache[key] = {'cnt': 1, 'val': value.astype(np.float32)}
            return



class _Program(object):
    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 p_cache=None,
                 verbose=0):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.verbose = verbose

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None
        self.p_cache: FactorValueCache = p_cache


    def build_program(self, random_state):
        """
        构建原生的随机表达式

        Parameters
        ----------
        random_state : RandomState instance
            随机状态生成

        Returns
        -------
        program : list
            展平的树结构的表达式（具体的解读方式类似于编译器对栈元素的解读）.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        # 确定了每次增加的最大深度
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        # 随机选择第一个初始的函数算子
        function = random_state.randint(len(self.function_set))
        function = deepcopy(self.function_set[function])

        program = [function]
        # 增加该算子需要的参数值
        terminal_stack = [function.arity]
        terminal_value_stack = []
        # 当terminal_stack 的参数值没有被填满，表达式不充实的时候
        while terminal_stack:
            # 查看栈的深度
            depth = len(terminal_stack)
            # 能选择的特征和其他算子总共有哪些
            choice = self.n_features + len(self.function_set)
            # 随机选择算子
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            # 决定了我们是增加一个新的函数还是算子
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = deepcopy(self.function_set[function])
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    # 生成特征还是常数
                    terminal = random_state.randint(self.n_features + 1)
                    # 如果说:
                    # 1. 当前的值是因子，且和之前的值相同，则重新生成因子
                    while True:
                        if (terminal in terminal_value_stack and terminal != self.n_features):
                            terminal = random_state.randint(self.n_features + 1)
                        else:
                            break

                else:
                    terminal = random_state.randint(self.n_features)

                if terminal == self.n_features:
                    # 这段代码确保了生成的 terminal 值既符合给定的范围，又不会是零，同时精确到小数点后三位
                    terminal = round(random_state.uniform(*self.const_range),3)
                    while True:
                        if terminal==0:
                            terminal = random_state.uniform(*self.const_range)
                        else:
                            break
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                if terminal_stack[-1]>0:
                    terminal_value_stack.append(terminal)
                while terminal_stack[-1] == 0:
                    terminal_value_stack = []
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        RandomFunctionStack = []
        # RandomFunctionStack[0].arity = 0
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                RandomFunctionStack.append(deepcopy(node))
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1

                if len(RandomFunctionStack)>0:
                    RandomFunctionStack[-1].arity -= 1
                    
                while terminals[-1] == 0:
                    RandomFunctionStack.pop()
                    terminals.pop()

                    terminals[-1] -= 1
                    if len(RandomFunctionStack)>0:
                        RandomFunctionStack[-1].arity -= 1
                        output += ')'
                    else:
                        output += ')'
                if i != len(self.program) - 1:
                    output += ', '

        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None
    
    def execute_3D(self, X):
        
        # Check for single-node programs
        node = self.program[0]

        if isinstance(node, float):
            return np.tile(node, (X.shape[0], X.shape[2]))
        if isinstance(node, int):
            return X[:, node, :]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                # print(apply_stack)
                # print(node.__str__())
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.tile(t, (X.shape[0],X.shape[2])) if isinstance(t, float) 
                             else X[:,t,:] if isinstance(t, int) 
                             else t for t in apply_stack[-1][1:]]
                
                # cache
                if self.p_cache:
                    tm = [t for t in apply_stack[-1][1:] if isinstance(t, int)]
                    if len(tm) == len(terminals):
                        k = f'{function.name}({", ".join([str(t) for t in tm])})'
                        intermediate_result = self.p_cache.get(k)
                        if not np.size(intermediate_result) > 0:
                            intermediate_result = function(*terminals)
                            self.p_cache.put(k, intermediate_result)
                    else:
                        self.p_cache.ttl += 1
                        intermediate_result = function(*terminals)
                else:
                    if self.p_cache:
                        self.p_cache.ttl += 1
                    intermediate_result = function(*terminals)

                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None
    

    def build_sub_expressions(self):
        
        # Check for single-node programs
        node = self.program[0]

        if isinstance(node, float):
            return []
        if isinstance(node, int):
            return []

        apply_stack = []
        sub_expr = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = apply_stack[-1][1:]
                tm = [t for t in apply_stack[-1][1:] if isinstance(t, int)]  # only take care of lowest level expression (ts_sum_5(10))
                intermediate_expr = f'{function.name}({", ".join([str(t) for t in terminals])})'
                if len(tm) == len(terminals):
                    sub_expr.append(intermediate_expr)

                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_expr)
                else:
                    return sub_expr

        # We should never get here
        return None



    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    def raw_fitness_3D(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        t0 = time.perf_counter()
        y_pred = self.execute_3D(X)
        cost_execute = time.perf_counter() - t0
        t1 = time.perf_counter()
        
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)
        
        cost_metric = time.perf_counter() - t1
        if self.verbose > 1:
            print(f'  [time cost] total: {time.perf_counter()-t0:.4f} execute_3d: {cost_execute:.4f} metric: {cost_metric:.4f} metric_val: {raw_fitness:.4f}')

        return raw_fitness


    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
