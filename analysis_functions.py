from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
import copy
from inspect import signature
from itertools import chain, combinations
from math import ceil, floor
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numbers
import numpy as np
import pandas as pd
import pickle
from sklearn.base import _pprint
from sklearn.metrics import precision_score, confusion_matrix, auc, average_precision_score, roc_auc_score 
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _approximate_mode
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils import resample
from sklearn.utils.validation import _num_samples, column_or_1d
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.fixes import comb
import random
from random import randrange, choice
import warnings

# from sklearn
class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for KFold, GroupKFold, and StratifiedKFold"""

    @abstractmethod
    def __init__(self, n_splits, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                .format(self.n_splits, n_samples))

        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class TimeSeriesSplitIgnoreSiblings(_BaseKFold):
    """Adapted from skearn TimeSeriesSplit. Effectively TimeSeriesSplit
    + GroupKFold.
    
    Description from sklearn.
    
    Time Series cross-validator
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    max_train_size : int, optional
        Maximum size for a single training set.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TimeSeriesSplit()
    >>> print(tscv)
    TimeSeriesSplit(max_train_size=None, n_splits=5)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    TRAIN: [0 1 2 3] TEST: [4]
    TRAIN: [0 1 2 3 4] TEST: [5]
    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i``th split,
    with a test set of size ``n_samples//(n_splits + 1)``,
    where ``n_samples`` is the number of samples.
    """
    def __init__(self, n_splits=5, max_train_size=None, sibling_group = None, sibling_na = 99999.0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.sibling_group = sibling_group
        self.sibling_na = sibling_na

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups  = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        sibling_group = self.sibling_group 
        sibling_na = self.sibling_na
        n_folds = n_splits + 1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        indices = np.arange(n_samples)
        test_size = (n_samples // n_folds)
        test_starts = range(test_size + n_samples % n_folds,
                            n_samples, test_size)

        for test_start in test_starts:
            train = list(indices[:test_start])
            test = list(indices[test_start:test_start + test_size])
            #print("Train indices before considering siblings: ", test_start)
            #print("Test indices before considering siblings: ", test_start, " : ", test_start + test_size)
            # Below is the adaptation
            train_siblings = sibling_group.loc[list(train)]
            test_siblings = sibling_group.loc[list(test)]
            siblings_in_both = list(set(test_siblings).intersection(set(train_siblings)))
            if sibling_na in siblings_in_both:
                siblings_in_both.remove(sibling_na)
            if siblings_in_both == []:
                # Check enough of each outcome in folds
                #print("Train classes: ", np.bincount(y[train]))
                #print("Test classes: ", np.bincount(y[test]))
                #print("Length of training data: ", len(train), "Length of test data:", len(test))
                yield (np.array(train), np.array(test))
            else:
                print("Number of siblings in both: ", len(siblings_in_both))
                sibling_group_to_drop_index = set(sibling_group.loc[sibling_group.isin(siblings_in_both)].index)
                # Can't get rid of siblings from the test dataset as test dataset becomes too small
                #print("Train indices: ", train)
                #print("Siblings to drop index: ", sibling_group_to_drop_index)
                train = [idx for idx in train if idx not in sibling_group_to_drop_index]
                #print(train)
                #Check enough of each outcome in folds
                #print(y[train])
                #print(y[test])      
                #print("Train classes: ", np.bincount(y[train]))
                #print("Test classes: ", np.bincount(y[test]))
                #print("Length of training data: ", len(train), "Length of test data:", len(test))
                train_siblings = sibling_group.loc[list(train)]
                test_siblings = sibling_group.loc[list(test)]
                siblings_in_both = list(set(test_siblings).intersection(set(train_siblings)))
                print("Siblings in both after: ", siblings_in_both)
                yield (np.array(train), np.array(test))

# from sklearn
class BaseShuffleSplit(metaclass=ABCMeta):
    """Base class for ShuffleSplit and StratifiedShuffleSplit"""

    def __init__(self, n_splits=10, test_size=None, train_size=None,
                 random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._default_test_size = 0.1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        X, y, groups = indexable(X, y, groups)
        for train, test in self._iter_indices(X, y, groups):
            yield train, test

    @abstractmethod
    def _iter_indices(self, X, y=None, groups=None):
        """Generate (train, test) indices"""

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def __repr__(self):
        return _build_repr(self)

# from sklearn, required in BaseShuffleSplit
def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values()
                       if p.name != 'self' and p.kind != p.VAR_KEYWORD])
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", FutureWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
                if value is None and hasattr(self, 'cvargs'):
                    value = self.cvargs.get(key, None)
            if len(w) and w[0].category == FutureWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))    
    

class StratifiedShuffleSplitGroups(BaseShuffleSplit):
    """Adapted from sklearn StratifiedShuffleSplit. Effectively 
    StratifiedShuffleSplit + GroupKFold.
    
    Blurb from sklearn:
    
    Stratified ShuffleSplit cross-validator
    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a merge of StratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class.
    Note: like the ShuffleSplit strategy, stratified random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default 10
        Number of re-shuffling & splitting iterations.
    test_size : float, int, None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.
    train_size : float, int, or None, default is None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    >>> sss.get_n_splits(X, y)
    5
    >>> print(sss)
    StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
    >>> for train_index, test_index in sss.split(X, y):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [5 2 3] TEST: [4 1 0]
    TRAIN: [5 1 4] TEST: [0 2 3]
    TRAIN: [5 0 2] TEST: [4 3 1]
    TRAIN: [4 1 0] TEST: [2 3 5]
    TRAIN: [0 5 1] TEST: [3 4 2]
    """

    def __init__(self, n_splits=10, test_size=None, train_size=None, random_state=None, 
                 sibling_group = None, sibling_na = 99999.0):
        
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)
        self._default_test_size = 0.1
        self.sibling_group = sibling_group
        self.sibling_na = sibling_na

    def _iter_indices(self, X, y, groups=None):
        sibling_group = self.sibling_group
        sibling_na = self.sibling_na
        n_samples = _num_samples(X)
        y = check_array(y, ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(
            n_samples, self.test_size, self.train_size,
            default_test_size=self._default_test_size)

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([' '.join(row.astype('str')) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)
        if np.min(class_counts) < 2:
            raise ValueError("The least populated class in y has only 1"
                             " member, which is too few. The minimum"
                             " number of groups for any class cannot"
                             " be less than 2.")

        if n_train < n_classes:
            raise ValueError('The train_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_train, n_classes))
        if n_test < n_classes:
            raise ValueError('The test_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_test, n_classes))

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(np.argsort(y_indices, kind='mergesort'),
                                 np.cumsum(class_counts)[:-1])

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)

            train = []
            test = []

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation,
                                                             mode='clip')

                train.extend(perm_indices_class_i[:n_i[i]])
                test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)
            
            train_siblings = sibling_group.loc[list(train)]
            test_siblings = sibling_group.loc[list(test)]
            siblings_in_both = list(set(test_siblings).intersection(set(train_siblings)))
            if sibling_na in siblings_in_both:
                siblings_in_both.remove(sibling_na)
            if siblings_in_both == []:
                # Check enough of each outcome in folds
                unique, counts = np.unique(y[train], return_counts=True)
                print("Train values: ", np.asarray((unique, counts)).T)
                unique, counts = np.unique(y[test], return_counts=True)
                print("Test values: ", np.asarray((unique, counts)).T)
                print("Length of training data: ", len(train), "Length of test data:", len(test))
                yield (np.array(train), np.array(test))
            else:
                #print("Number of siblings in both: ", len(siblings_in_both))
                train_siblings_in_both = siblings_in_both[:int(len(siblings_in_both)/2)]
                #print(train_siblings_in_both)
                test_siblings_in_both = siblings_in_both[int(len(siblings_in_both)/2):]
                #print(test_siblings_in_both)
                sibling_group_to_drop_train_index = list(sibling_group.loc[sibling_group.isin(test_siblings_in_both)].index)
                #print(sibling_group_to_drop_train_index)
                sibling_group_to_drop_test_index = list(sibling_group.loc[sibling_group.isin(train_siblings_in_both)].index)
                #print(sibling_group_to_drop_test_index)
                train = [idx for idx in train if idx not in sibling_group_to_drop_train_index]
                test = [idx for idx in test if idx not in sibling_group_to_drop_test_index]
                print("Check intersection train / test: ", set(train).intersection(set(test)))
                # Check enough of each outcome in folds
                print("Train classes: ", np.bincount(y_indices[train]))
                print("Test classes: ", np.bincount(y_indices[test]))
                print("Length of training data: ", len(train), "Length of test data:", len(test))
                train_siblings = sibling_group.loc[list(train)]
                test_siblings = sibling_group.loc[list(test)]
                siblings_in_both = list(set(test_siblings).intersection(set(train_siblings)))
                print("Siblings in both after: ", siblings_in_both)
                #print("Child_IDs in both after: ", list(set(X.loc[train, 'Child_ID']).intersection(set(X.loc[test, 'Child_ID']))))
                yield (np.array(train), np.array(test))

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)

# from sklearn
def _validate_shuffle_split(n_samples, test_size, train_size,
                            default_test_size=None):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (test_size_type == 'i' and (test_size >= n_samples or test_size <= 0)
       or test_size_type == 'f' and (test_size <= 0 or test_size >= 1)):
        raise ValueError('test_size={0} should be either positive and smaller'
                         ' than the number of samples {1} or a float in the '
                         '(0, 1) range'.format(test_size, n_samples))

    if (train_size_type == 'i' and (train_size >= n_samples or train_size <= 0)
       or train_size_type == 'f' and (train_size <= 0 or train_size >= 1)):
        raise ValueError('train_size={0} should be either positive and smaller'
                         ' than the number of samples {1} or a float in the '
                         '(0, 1) range'.format(train_size, n_samples))

    if train_size is not None and train_size_type not in ('i', 'f'):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ('i', 'f'):
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    if (train_size_type == 'f' and test_size_type == 'f' and
            train_size + test_size > 1):
        raise ValueError(
            'The sum of test_size and train_size = {}, should be in the (0, 1)'
            ' range. Reduce test_size and/or train_size.'
            .format(train_size + test_size))

    if test_size_type == 'f':
        n_test = ceil(test_size * n_samples)
    elif test_size_type == 'i':
        n_test = float(test_size)

    if train_size_type == 'f':
        n_train = floor(train_size * n_samples)
    elif train_size_type == 'i':
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n_samples))

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            'With n_samples={}, test_size={} and train_size={}, the '
            'resulting train set will be empty. Adjust any of the '
            'aforementioned parameters.'.format(n_samples, test_size,
                                                train_size)
        )

    return n_train, n_test


def visualize_groups(classes, groups, name, cmap_data, cmap_cv):
    '''Adapted from: 
    
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html'''
    # Visualize dataset groups
    fig, ax = plt.subplots()
    # c needs to be a number or colour list
    if any(isinstance(val, str) for val in groups):
        groups, _ = pd.factorize(groups)
    ax.scatter(range(len(groups)),  [.5] * len(groups), c=groups, marker='_',
               lw=50, cmap=cmap_data)
    ax.scatter(range(len(groups)),  [3.5] * len(groups), c=classes, marker='_',
               lw=50, cmap=cmap_data)
    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['Data\ngroup', 'Data\nclass'], xlabel="Sample index")

def plot_cv_indices(cv, X, y, group, ax, n_splits, cmap_cv, cmap_data, lw=10):
    '''Create a sample plot for indices of a cross-validation object.
     Adapted from: 
    
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html'''
    # c needs to be a number or colour list
    if any(isinstance(val, str) for val in group):
        group, _ = pd.factorize(group)
        
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(X)), [ii + .5] * len(X),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)
    
    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(X)])
    return ax

def create_test_train_holdout_splits(df, splitter, splitter_name, outcome, siblings, suffix):
    '''for ss splitter should have 1 split to split into train and test / holdout. Then we split test and holdout.
    Can't use the same approach for ts because it expects splits > 2 and doesn't allow setting of test_size.'''
    # Split data into features and outcome
    X = df.drop(columns = outcome)
    y = df[outcome]
    # Generate list of indices
    print("Using splitter")
    train_test_list, len_train_test_list = [], []
    for train_index, test_index in splitter.split(X, y): 
        len_train_test_list.append((len(train_index), len(test_index)))
        train_test_list.append((train_index, test_index))
    # Split by index
    if splitter_name == 'ts':
        X_test = X.loc[train_test_list[3][1],]
        y_test = y.loc[train_test_list[3][1],]
        X_holdout = X.loc[train_test_list[4][1],]
        y_holdout = y.loc[train_test_list[4][1],]
        train_indices = set(train_test_list[3][0]).intersection(set(train_test_list[4][0]))
        X_tr = X.loc[train_indices,]
        y_tr = y.loc[train_indices]
    if splitter_name == 'ss':
        print("Split by index")
        X_tr = X.loc[train_test_list[0][0],]
        test_indices = train_test_list[0][1][:floor(len(train_test_list[0][1])/2)]
        holdout_indices = train_test_list[0][1][ceil(len(train_test_list[0][1])/2):]
        X_test = X.loc[test_indices,]
        X_holdout = X.loc[holdout_indices,]
        # Create y
        y_tr = y.loc[X_tr.index]
        y_test = y.loc[X_test.index]
        y_holdout = y.loc[X_holdout.index]
    # Print to check values
    print(X_tr.shape)
    print(X_test.shape)
    print(X_holdout.shape)
    print(y_tr.shape)
    print("y_tr values: ", y_tr.value_counts())
    print(y_test.shape)
    print("y_test values: ", y_test.value_counts())
    print(y_holdout.shape)
    print("y_holdout values: ", y_holdout.value_counts())
    # Check siblings
    siblings_train = siblings.loc[X_tr.index]
    siblings_test = siblings.loc[X_test.index]
    siblings_holdout = siblings.loc[X_holdout.index]
    # Check no overlap in siblings between train, test and holdout data
    print("Number of overlapping siblings in train / test: ", len(set(siblings_train).intersection(set(siblings_test))))
    print(set(siblings_train).intersection(set(siblings_test)))
    print("Number of overlapping siblings in train / holdout: ", len(set(siblings_train).intersection(set(siblings_holdout))))
    print(set(siblings_train).intersection(set(siblings_holdout)))
    # Save files
    print("Saving")
    with open("../../Data for Model/X_train_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(X_tr, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/X_test_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(X_test, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/X_holdout_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(X_holdout, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/y_train_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(y_tr, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/y_test_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(y_test, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/y_holdout_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(y_holdout, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/siblings_train_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(siblings_train, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/siblings_test_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(siblings_test, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/siblings_holdout_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(siblings_holdout, handle, protocol = pickle.HIGHEST_PROTOCOL)

        
def create_test_train_splits(df, splitter, splitter_name, outcome, siblings, suffix):
    '''Same as above but without holdout dataset.'''
    # Split data into features and outcome
    X = df.drop(columns = outcome)
    y = df[outcome]
    # Generate list of indices
    print("Using splitter")
    train_test_list, len_train_test_list = [], []
    for train_index, test_index in splitter.split(X, y): 
        len_train_test_list.append((len(train_index), len(test_index)))
        train_test_list.append((train_index, test_index))
    # Split by index
    if splitter_name == 'ts':
        X_test = X.loc[train_test_list[splitter.n_splits-1][1],] # just 20% used as test data
        y_test = y.loc[train_test_list[splitter.n_splits-1][1],]
        train_indices = set(train_test_list[splitter.n_splits-1][0])
        X_tr = X.loc[train_indices,]
        y_tr = y.loc[train_indices]
    if splitter_name == 'ss':
        print("Split by index")
        X_tr = X.loc[train_test_list[0][0],]
        X_test = X.loc[train_test_list[0][1],]
        # Create y
        y_tr = y.loc[X_tr.index]
        y_test = y.loc[X_test.index]
    # Print to check values
    print(X_tr.shape)
    print(X_test.shape)
    print(y_tr.shape)
    print("y_tr values: ", y_tr.value_counts())
    print(y_test.shape)
    print("y_test values: ", y_test.value_counts())
    # Check siblings
    siblings_train = siblings.loc[X_tr.index]
    siblings_test = siblings.loc[X_test.index]
    # Check no overlap in siblings between train, test and holdout data
    print("Number of overlapping siblings in train / test: ", len(set(siblings_train).intersection(set(siblings_test))))
    print(set(siblings_train).intersection(set(siblings_test)))
    # Save files
    print("Saving")
    with open("../../Data for Model/X_train_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(X_tr, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/X_test_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(X_test, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/y_train_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(y_tr, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/y_test_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(y_test, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/siblings_train_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(siblings_train, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open("../../Data for Model/siblings_test_{}.pkl".format(suffix), "wb") as handle:
        pickle.dump(siblings_test, handle, protocol = pickle.HIGHEST_PROTOCOL)        
        
def grid_search_save_output(gscv, algorithm, rcv_n_iter, parameters, X, y, file_name):
    gscv.fit(X, y)
    print("Best parameters: ", gscv.best_params_)
    print("Best estimator", gscv.best_estimator_)
    print("Mean of best estimator (ts):", gscv.cv_results_['mean_test_score'][gscv.best_index_])
    print("Std of best estimator (ts):",  gscv.cv_results_['std_test_score'][gscv.best_index_])
    # Creating a list of model dictionaries
    df = pd.DataFrame(gscv.cv_results_)
    df_params = df['params'].apply(pd.Series)
    df.drop(columns = 'params', inplace = True)
    df_all = pd.concat([df, df_params], axis = 1)
    df_all['algorithm'] = algorithm
    df_all['rcv_n_iter'] = rcv_n_iter
    df_all['parameter_combination'] = parameters
    df_all.to_csv(file_name)
    return(df_all, gscv.best_params_, gscv.best_estimator_)

# For feature importance - getting column names from transformation on text
def get_column_names_from_ColumnTransformer(column_transformer):    
    col_name = []
    for transformer_in_columns in column_transformer.transformers_[:-1]:#the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1],Pipeline): 
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except(AttributeError): # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names,np.ndarray): # eg.
            col_name += names.tolist()
        elif isinstance(names,list):
            col_name += names    
        elif isinstance(names,str):
            col_name.append(names)
    return col_name

def pinned_metrics(df, group, subgroups, y_true, y_scores, y_proba_scores, metric, upsampling = True):
    '''
    parameters:
        df = dataframe of observations x  columns (subgroup, y_true, y_scores)
        group = column name containing the subgroup of interest (can be multiple categories)
        subgroups = subgroup(s) of interest (can be iterable)
        y_true = column name of binary outcome variable
        y_scores = column name of prediction
        metric = average_precision_score and roc_auc_score (as per sklearn), or:

            false_discovery_rate = FP / (FP + TP)
            false_omission_rate = FN / (FN + TN)

        upsampling = if True (default) then upsample the minority group; if False then score on subgroup in data
        
    returns:
        if subgroups is iterable, then returns a list of metrics of len(subgroups); otherwise returns metric
    '''
    def pinned_metric_for_single_subgroup(df = df, group = group, subgroups = subgroups, y_true = y_true, y_scores = y_scores, y_proba_scores = y_proba_scores, metric = metric, upsampling = True):
        df = copy.deepcopy(df)
        df['subgroup'] = [1 if s==subgroup else 0 for s in df[group]]
        df_majority = df.loc[df['subgroup']==0,]
        df_minority = df.loc[df['subgroup']==1,]
        
        # If df_minority is not empty:
        if df_minority.shape[0] != 0:
            
            # Upsample minority class (also downsamples majority class)
            if upsampling == True:
                df_minority_upsampled = resample(df_minority, 
                                              replace=True,     # sample with replacement
                                              n_samples=df_majority.shape[0], # to match majority class
                                              random_state=3005) # reproducible results

                # Combine majority class with upsampled minority class
                df_for_calc = pd.concat([df_majority, df_minority_upsampled])

            # Calculate metric just on subgroup
            else:
                df_for_calc = df_minority

            # Score
            if metric == 'average_precision_score':
                pinned_metric = average_precision_score(df_for_calc[y_true], df_for_calc[y_proba_scores])
            if metric == 'roc_auc_score':
                pinned_metric = roc_auc_score(df_for_calc[y_true], df_for_calc[y_proba_scores])
            if metric == 'false_discovery_rate':
                precision = precision_score(df_for_calc[y_true], df_for_calc[y_scores])
                pinned_metric = 1-precision
            if metric == 'false_omission_rate':
                tn, fp, fn, tp = confusion_matrix(df_for_calc[y_true], df_for_calc[y_scores]).ravel()
                pinned_metric = fn / (fn + tn)
            return pinned_metric
        else:
            return np.nan

    if isinstance(subgroups, Iterable):
        pinned_metrics = []
        for subgroup in subgroups:
            pinned_metrics.append(pinned_metric_for_single_subgroup())
        return pinned_metrics
    else:
        return pinned_metric_for_single_subgroup()

def create_synethetic_structured_data(T, N, k):
    """
    Basically recreates SMOTE!
    
    Returns (N/100) * n_minority_samples synthetic minority samples.
    
    Returned synetheic observation is a random choice of a nearest
    neighbour with added random noise.
    
    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 

    Returns
    -------
    S : array, shape = [(N/100) * n_minority_samples, n_features]
    """    
    n_minority_samples, n_features = T.shape

    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")

    N = int(N/100)
    n_synthetic_samples = int(N * n_minority_samples)
    S = np.zeros(shape=(n_synthetic_samples, n_features))

    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)

    #Calculate synthetic samples
    for i in range(0, n_minority_samples):
        nn = neigh.kneighbors([T[i]], return_distance=False)
        for n in range(N):
            # nn[0] is the index of the nearest neighbours in order, of length n_neighbors
            # choice randomly picks one
            nn_index = choice(nn[0]) 
            # NOTE: nn includes T[i], we don't want to select it 
            while nn_index == i:
                nn_index = choice(nn[0])
            # Adds noise
            dif = T[nn_index,:] - T[i,:]
            gap = np.random.random()
            S[n + i * N, :] = T[i,:] + gap * dif[:]

    return S