# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 08:59:01 2018

@author: Lionel Massoulard
"""


from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import TruncatedSVD

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from aikit.transformers.model_wrapper import ModelWrapper


import numpy as np


# In[]


def compute_node_norm_classification_tree(tree):
    """ takes a DecisionTree Regressor and returns a value corresponding to the norm of each node, as well the coefficient of each node """
    value = tree.tree_.value

    ch_left = tree.tree_.children_left
    ch_right = tree.tree_.children_right
    nb_nodes = tree.tree_.node_count

    parents = -np.ones(nb_nodes, dtype=np.int32)
    nodes_index = np.arange(nb_nodes)

    parents[ch_left[ch_left != -1]] = nodes_index[ch_left != -1]
    parents[ch_right[ch_right != -1]] = nodes_index[ch_right != -1]

    sum_v = value.sum(axis=2, keepdims=True)
    proba = value / sum_v

    ii = parents != -1
    ii_root = parents == -1

    nodes_value = np.zeros(proba.shape, dtype=np.float32)

    nodes_value[ii] = proba[ii, :, :] - proba[parents[ii], :, :]
    nodes_value[ii_root] = proba[ii_root, :, :]

    delta_norm = (nodes_value ** 2).sum(axis=2).sum(axis=1)

    nodes_norm = sum_v[:, 0, 0] * delta_norm

    return nodes_norm, nodes_value


def compute_node_norm_regression_tree(tree):
    """ takes a DecisionTree Classifier and returns a value corresponding to the norm of each node, as well the coefficient of each node """

    ch_left = tree.tree_.children_left
    ch_right = tree.tree_.children_right

    value = tree.tree_.value
    n_node_samples = tree.tree_.n_node_samples

    nb_nodes = tree.tree_.node_count

    parents = -np.ones(nb_nodes, dtype=np.int32)
    nodes_index = np.arange(nb_nodes)

    parents[ch_left[ch_left != -1]] = nodes_index[ch_left != -1]
    parents[ch_right[ch_right != -1]] = nodes_index[ch_right != -1]

    ii = parents != -1
    ii_root = parents == -1

    nodes_value = np.zeros(value.shape, dtype=np.float32)

    nodes_value[ii] = value[ii, :, :] - value[parents[ii], :, :]
    nodes_value[ii_root] = value[ii_root, :, :]

    delta_norm = (nodes_value ** 2).sum(axis=2).sum(axis=1)

    nodes_norm = n_node_samples * delta_norm

    return nodes_norm, nodes_value


def compute_node_dept_is_leaves(tree):
    """ takes a Decision Tree and returns information about each nodes : depts and if it a leaf or not """

    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    nodes_depth = np.zeros(shape=n_nodes, dtype=np.int32)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        nodes_depth[node_id] = parent_depth + 1

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return nodes_depth, is_leaves


def compute_node_norm_regression_forest(forest):

    all_nodes_norms = []
    all_nodes_values = []

    for tree in forest.estimators_:
        node_norm, delta_value = compute_node_norm_regression_tree(tree)

        all_nodes_norms.append(node_norm)
        all_nodes_values.append(delta_value)

    forest_nodes_norm = np.concatenate(all_nodes_norms, axis=0)
    forest_nodes_value = np.concatenate(all_nodes_values, axis=0)

    forest_nodes_value /= len(forest.estimators_)

    return forest_nodes_norm, forest_nodes_value


def compute_node_norm_classification_forest(forest):

    all_nodes_norms = []
    all_nodes_values = []

    for tree in forest.estimators_:
        nodes_norm, nodes_value = compute_node_norm_classification_tree(tree)

        all_nodes_norms.append(nodes_norm)
        all_nodes_values.append(nodes_value)

    forest_nodes_norm = np.concatenate(all_nodes_norms, axis=0)
    forest_nodes_value = np.concatenate(all_nodes_values, axis=0)

    forest_nodes_value /= len(forest.estimators_)

    return forest_nodes_norm, forest_nodes_value


class WaveRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    RandomForest based classifier but with nodes that are removed
    
    See Paper:
    Wavelet decomposition of Random Forests
    http://www.jmlr.org/papers/volume17/15-203/15-203.pdf
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        nodes_to_keep=0.9,
    ):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

        self.nodes_to_keep = nodes_to_keep

        self.forest = None

    def fit(self, X, y):

        # 1) create RandomForest
        self.forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
        )

        # 2) fit it
        self.forest.fit(X, y)

        self.n_outputs_ = self.forest.n_outputs_

        # 3) retrieve node norms and values
        self.nodes_norm, self.nodes_value = compute_node_norm_classification_forest(self.forest)

        # 4) filter nodes
        self._nodes_order = np.argsort(-self.nodes_norm)

        if self.nodes_to_keep is not None:
            if self.nodes_to_keep < 1:
                nodes_to_keep = int(len(self._nodes_order) * self.nodes_to_keep)
            else:
                nodes_to_keep = int(self.nodes_to_keep)

            self._ind_nodes_to_keep = self._nodes_order[:nodes_to_keep]
        else:
            self._ind_nodes_to_keep = None

        return self

    def _set_nodes_to_keep(self, nodes_to_keep):
        """ change the number of waweletts to keep withtout refitting the underlying random forest """
        self.nodes_to_keep = nodes_to_keep

        if self.forest is not None:

            if self.nodes_to_keep is None:
                self._ind_nodes_to_keep = None

            else:
                if self.nodes_to_keep < 1:
                    nodes_to_keep = int(len(self._nodes_order) * self.nodes_to_keep)
                else:
                    nodes_to_keep = int(self.nodes_to_keep)

            self._ind_nodes_to_keep = self._nodes_order[:nodes_to_keep]

    def predict_proba(self, X):

        if self.forest is None:
            raise NotFittedError("You should fit the model first")

        path, _ = self.forest.decision_path(X)

        if self._ind_nodes_to_keep is not None:
            predict_proba_filtered = [
                path[:, self._ind_nodes_to_keep].dot(self.nodes_value[self._ind_nodes_to_keep, n, :])
                for n in range(self.nodes_value.shape[1])
            ]
        else:
            predict_proba_filtered = [
                path[:, :].dot(self.nodes_value[:, n, :]) for n in range(self.nodes_value.shape[1])
            ]

        for p in predict_proba_filtered:
            p[p < 0] = 0
            p[p > 1] = 1

        if len(predict_proba_filtered) == 1:
            return predict_proba_filtered[0]
        else:
            return predict_proba_filtered

    @property
    def classes_(self):
        return self.forest.classes_

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        # Copied from base forest
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            predictions = np.zeros((n_samples, self.n_outputs_))

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k], axis=1), axis=0)

            return predictions

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        # Copied from base forest
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class WaveRandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    RandomForest based classifier but with nodes that are removed
    
    See Paper:
    Wavelet decomposition of Random Forests
    http://www.jmlr.org/papers/volume17/15-203/15-203.pdf
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        nodes_to_keep=0.9,
    ):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.nodes_to_keep = nodes_to_keep

        self.forest = None

    def fit(self, X, y):

        # 1) create RandomForest
        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
        )

        # 2) fit it
        self.forest.fit(X, y)

        self.n_outputs_ = self.forest.n_outputs_

        # 3) retrieve node norms and values
        self.nodes_norm, self.nodes_value = compute_node_norm_regression_forest(self.forest)

        # 4) filter nodes
        self._nodes_order = np.argsort(-self.nodes_norm)

        if self.nodes_to_keep is not None:
            if self.nodes_to_keep < 1:
                nodes_to_keep = int(len(self._nodes_order) * self.nodes_to_keep)
            else:
                nodes_to_keep = int(self.nodes_to_keep)

            self._ind_nodes_to_keep = self._nodes_order[:nodes_to_keep]
        else:
            self._ind_nodes_to_keep = None

        return self

    def _set_nodes_to_keep(self, nodes_to_keep):
        """ change the number of waweletts to keep withtout refitting the underlying random forest """
        self.nodes_to_keep = nodes_to_keep

        if self.forest is not None:

            if self.nodes_to_keep is None:
                self._ind_nodes_to_keep = None

            else:
                if self.nodes_to_keep < 1:
                    nodes_to_keep = int(len(self._nodes_order) * self.nodes_to_keep)
                else:
                    nodes_to_keep = int(self.nodes_to_keep)

            self._ind_nodes_to_keep = self._nodes_order[:nodes_to_keep]

    def predict(self, X):

        if self.forest is None:
            raise NotFittedError("You should fit the model first")

        path, _ = self.forest.decision_path(X)

        if self._ind_nodes_to_keep is not None:
            predict_proba_filtered = [
                path[:, self._ind_nodes_to_keep].dot(self.nodes_value[self._ind_nodes_to_keep, n, :])
                for n in range(self.nodes_value.shape[1])
            ]
        else:
            predict_proba_filtered = [
                path[:, :].dot(self.nodes_value[:, n, :]) for n in range(self.nodes_value.shape[1])
            ]

        if len(predict_proba_filtered) == 1:
            return predict_proba_filtered[0][:, 0]
        else:
            return predict_proba_filtered


# In[]


class _RandomForestLinear(BaseEstimator, ClassifierMixin):
    """ This model is a mixture of a classical RandomForest with on linear model plug after it
    The idea is to fit a RandomForest and use the node as features for a linear model.
    So re-optimizing globally the structure created by the RandomForest
    
    Parameters
    ----------
    
    n_estimators : int, default = 100
        number of trees of the RandomForest
        
    criterion : string, default = 'gini' or 'mse'
        the splitting criterion for the RandomForest
        
    max_deatures : string or number, default = 'auto',
        the number of features per split
        
    max_depth : int or None, default = None
        the maximum depth of trees
        
    random_state : int or None
        random seed for RandomForest
        
    other_rf_params : dict or None
        additionnal parameters to be passed to the RandomForest
    
    do_svd : boolean, default = False
        if True will do an SVD before calling the linear algorithm
        
    svd_n_components : int, default = 100
        number of svd components
    
    C : float, default = 1
        linear model C parameter
        
    """

    is_regression = None

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_features="auto",
        max_depth=None,
        random_state=None,
        nodes_to_keep=None,
        other_rf_params=None,
        do_svd=False,
        svd_n_components=100,
        C=1,
    ):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.do_svd = do_svd
        self.svd_n_components = svd_n_components

        self.nodes_to_keep = nodes_to_keep

        self.other_rf_params = other_rf_params

        self.C = C

    def fit(self, X, y=None):

        if self.is_regression:
            rf_klass = RandomForestRegressor
            lin_klass = Ridge
            kwargs = {"alpha": self.C}
        else:
            rf_klass = RandomForestClassifier
            lin_klass = LogisticRegression
            kwargs = {"C": self.C}

        if self.other_rf_params is None:
            other_rf_params = {}
        else:
            other_rf_params = self.other_rf_params

        self.forest = rf_klass(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=self.max_features,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **other_rf_params
        )

        self.forest.fit(X, y)

        Xnode_onehot, _ = self.forest.decision_path(X)

        # Filter of Nodes ?
        if self.nodes_to_keep is not None:

            if self.is_regression:
                nodes_norm, nodes_value = compute_node_norm_regression_forest(self.forest)
            else:
                nodes_norm, nodes_value = compute_node_norm_regression_forest(self.forest)

            nodes_order = np.argsort(-nodes_norm)

            if self.nodes_to_keep < 1:
                nodes_to_keep = int(len(nodes_order) * self.nodes_to_keep)
            else:
                nodes_to_keep = int(self.nodes_to_keep)

            self._ind_nodes_to_keep = nodes_order[:nodes_to_keep]

            Xnode_onehot = Xnode_onehot[:, self._ind_nodes_to_keep]

        else:
            self._ind_nodes_to_keep = None

        if self.do_svd:
            self.svd = TruncatedSVD(n_components=100)
            Xsvd = self.svd.fit_transform(Xnode_onehot)
        else:
            Xsvd = Xnode_onehot

        self.linear = lin_klass(**kwargs)

        self.linear.fit(Xsvd, y)

        return self

    def predict(self, X):

        Xnode_onehot, _ = self.forest.decision_path(X)

        if self._ind_nodes_to_keep is not None:
            Xnode_onehot = Xnode_onehot[:, self._ind_nodes_to_keep]

        if self.do_svd:
            Xsvd = self.svd.transform(Xnode_onehot)
        else:
            Xsvd = Xnode_onehot

        return self.linear.predict(Xsvd)


class RandomForestLogit(_RandomForestLinear):
    __doc__ = _RandomForestLinear.__doc__

    is_regression = False

    @property
    def classes_(self):
        return self.linear.classes_

    def predict_proba(self, X):

        Xnode_onehot, _ = self.forest.decision_path(X)

        if self._ind_nodes_to_keep is not None:
            Xnode_onehot = Xnode_onehot[:, self._ind_nodes_to_keep]

        if self.do_svd:
            Xsvd = self.svd.transform(Xnode_onehot)
        else:
            Xsvd = Xnode_onehot

        return self.linear.predict_proba(Xsvd)

    def predict_log_proba(self, X):

        Xnode_onehot, _ = self.forest.decision_path(X)

        if self._ind_nodes_to_keep is not None:
            Xnode_onehot = Xnode_onehot[:, self._ind_nodes_to_keep]

        if self.do_svd:
            Xsvd = self.svd.transform(Xnode_onehot)
        else:
            Xsvd = Xnode_onehot

        return self.linear.predict_log_proba(Xsvd)


class RandomForestRidge(_RandomForestLinear):
    __doc__ = _RandomForestLinear.__doc__

    is_regression = True

    def __init__(
        self,
        n_estimators=100,
        criterion="mse",  # change default argument
        max_features="auto",
        max_depth=None,
        random_state=None,
        nodes_to_keep=None,
        other_rf_params=None,
        do_svd=False,
        svd_n_components=100,
        C=1,
    ):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.nodes_to_keep = nodes_to_keep
        self.do_svd = do_svd
        self.svd_n_components = svd_n_components
        self.other_rf_params = other_rf_params
        self.C = C


# In[]


# In[]


class _RandomForestTransformerAbstract(BaseEstimator, TransformerMixin):
    """ This model is a transforms a classical RandomForest into a transformer by returning not the prediction but the nodes.
    
    The process is the following :
        1. fit a RandomForest
        2. get the node dummy variable (using decision path)
        3. (optional) filter some of the nodes
        4. (optional) apply an SVD
        
    It can be useful to
    * craft non-linear features that can be given to a linear algorithm
    * create a 'supervised' clustering algorithm
    * create a similarity between observations based on their nodes
    * ...
        

    Parameters
    ----------
    
    n_estimators : int, default = 100
        number of trees of the RandomForest
        
    criterion : string, default = 'gini' or 'mse'
        the splitting criterion for the RandomForest
        
    max_deatures : string or number, default = 'auto',
        the number of features per split
        
    max_depth : int or None, default = None
        the maximum depth of trees
        
    random_state : int or None
        random seed for RandomForest
        
    nodes_to_keep : int, float or None
        number of nodes to keep in result (filter by their norm), if None no filter, if float < 1 taken as a percentage of the total number of nodes
        
    other_rf_params : dict or None
        additionnal parameters to be passed to the RandomForest
    
    do_svd : boolean, default = False
        if True will do an SVD before calling the linear algorithm
        
    svd_n_components : int, default = 100
        number of svd components
    
    """

    is_regression = None

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_features="auto",
        max_depth=None,
        random_state=None,
        nodes_to_keep=None,
        other_rf_params=None,
        do_svd=False,
        svd_n_components=100,
    ):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.nodes_to_keep = nodes_to_keep
        self.do_svd = do_svd
        self.svd_n_components = svd_n_components
        self.other_rf_params = other_rf_params

    def fit(self, X, y):
        self._fit_transform(X, y, do_fit=True, do_transform=False)
        return self

    def transform(self, X):
        Xres = self._fit_transform(X, y=None, do_fit=False, do_transform=True)
        return Xres

    def fit_transform(self, X, y):
        Xres = self._fit_transform(X, y, do_fit=True, do_transform=True)
        return Xres

    def _fit_transform(self, X, y, do_fit, do_transform):

        if do_fit:
            if self.other_rf_params is None:
                other_rf_params = {}
            else:
                other_rf_params = self.other_rf_params

            if self.is_regression:
                rf_klass = RandomForestRegressor
            else:
                rf_klass = RandomForestClassifier

            ## 1) create RF and fit it
            self.forest = rf_klass(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_features=self.max_features,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **other_rf_params
            )
            self.forest.fit(X, y)

        ## 2) retrieve node id
        Xnode_onehot, _ = self.forest.decision_path(X)

        ### 3) filter nodes
        if do_fit:
            if self.nodes_to_keep is not None:

                if self.is_regression:
                    nodes_norm, nodes_value = compute_node_norm_regression_forest(self.forest)
                else:
                    nodes_norm, nodes_value = compute_node_norm_regression_forest(self.forest)

                nodes_order = np.argsort(-nodes_norm)

                if self.nodes_to_keep < 1:
                    nodes_to_keep = int(len(nodes_order) * self.nodes_to_keep)
                else:
                    nodes_to_keep = int(self.nodes_to_keep)

                self._ind_nodes_to_keep = nodes_order[:nodes_to_keep]

            else:
                self._ind_nodes_to_keep = None

        if self._ind_nodes_to_keep is not None:
            Xnode_onehot = Xnode_onehot[:, self._ind_nodes_to_keep]

        if self.do_svd:
            if do_fit:
                self.svd = TruncatedSVD(n_components=self.svd_n_components)
                Xsvd = self.svd.fit_transform(Xnode_onehot)
            else:
                Xsvd = self.svd.transform(Xnode_onehot)
        else:
            Xsvd = Xnode_onehot

        if do_fit:
            if self.do_svd:
                self._features_names = ["RFNODE_SVD_%d" % i for i in range(Xsvd.shape[1])]
            else:
                self._features_names = ["RFNODE_%d" % i for i in range(Xsvd.shape[1])]

        if do_transform:
            return Xsvd
        else:
            return self

    def get_feature_names(self):
        return self._features_names


class _RandomForestClassifierTransformer(_RandomForestTransformerAbstract):
    __doc__ = _RandomForestTransformerAbstract.__doc__

    is_regression = False


class _RandomForestRegressorTransformer(_RandomForestTransformerAbstract):
    __doc__ = _RandomForestTransformerAbstract.__doc__

    is_regression = True


class RandomForestClassifierTransformer(ModelWrapper):
    __doc__ = _RandomForestTransformerAbstract.__doc__

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_features="auto",
        max_depth=None,
        random_state=None,
        nodes_to_keep=None,
        do_svd=False,
        svd_n_components=100,
        other_rf_params=None,
        columns_to_use="all",
        desired_output_type=None,
    ):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.nodes_to_keep = nodes_to_keep

        self.do_svd = do_svd
        self.svd_n_components = svd_n_components
        self.other_rf_params = other_rf_params

        self.columns_to_use = columns_to_use
        self.desired_output_type = desired_output_type

        super(RandomForestClassifierTransformer, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=False,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
        )

    def _get_model(self, X, y=None):
        return _RandomForestClassifierTransformer(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=self.max_features,
            random_state=self.random_state,
            nodes_to_keep=self.nodes_to_keep,
            do_svd=self.do_svd,
            svd_n_components=self.svd_n_components,
            other_rf_params=self.other_rf_params,
        )


class RandomForestRegressorTransformer(ModelWrapper):
    __doc__ = _RandomForestTransformerAbstract.__doc__

    def __init__(
        self,
        n_estimators=100,
        criterion="mse",
        max_features="auto",
        max_depth=None,
        random_state=None,
        nodes_to_keep=None,
        do_svd=False,
        svd_n_components=100,
        other_rf_params=None,
        columns_to_use="all",
        desired_output_type=None,
    ):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.nodes_to_keep = nodes_to_keep
        self.do_svd = do_svd
        self.svd_n_components = svd_n_components
        self.other_rf_params = other_rf_params

        self.columns_to_use = columns_to_use
        self.desired_output_type = desired_output_type

        super(RandomForestRegressorTransformer, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=False,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
        )

    def _get_model(self, X, y=None):
        return _RandomForestRegressorTransformer(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=self.max_features,
            random_state=self.random_state,
            nodes_to_keep=self.nodes_to_keep,
            do_svd=self.do_svd,
            svd_n_components=self.svd_n_components,
            other_rf_params=self.other_rf_params,
        )
