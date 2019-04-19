# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:35:25 2018

@author: Lionel Massoulard
"""


import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin, RegressorMixin

from sklearn.ensemble.forest import ForestClassifier, ForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree.tree import DTYPE


from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, check_array
from sklearn.decomposition import PCA

from scipy.sparse import issparse


from aikit.tools.data_structure_helper import get_type, DataTypes


class GroupPCA(TransformerMixin, BaseEstimator):
    """ PCAs on random group of features
    
    Parameters
    ----------
    random_state : int or None, default = None
        the state of random generator
        
    bootstrap : boolean, default = False
        if True will fit PCA on random subset of rows
        
    max_nb_groups : int or float
        the maximum number of groups of features (if <1 percentage of all features)
        
    max_group_size : int or float
        the maximum number of features per group (if <1 percentage of all features)
    
    """

    def __init__(self, random_state=None, bootstrap=False, max_nb_groups=0.25, max_group_size=0.05):

        self.random_state = random_state
        self.bootstrap = bootstrap

        self.max_nb_groups = max_nb_groups
        self.max_group_size = max_group_size

        self._scaler = None
        self._components_ = None

    def fit(self, X, y=None, **fit_params):

        # * save type
        self._input_type = get_type(X)
        self._nb_cols = X.shape[1]

        NF = X.shape[1]

        # * scale features
        self._scaler = StandardScaler(
            with_mean=self._input_type not in (DataTypes.SparseArray, DataTypes.SparseDataFrame)
        )
        Xz = self._scaler.fit_transform(X)

        # * random generator
        random_state = check_random_state(self.random_state)

        # Number of splits
        if self.max_nb_groups < 1:
            high = int(NF * self.max_nb_groups)
        else:
            high = min(int(self.max_nb_groups), NF - 1)

        self._nb_of_groups = random_state.randint(low=1, high=high)

        # all splits
        if self.max_group_size < 1:
            high_f = max(int(NF * self.max_group_size), 5)
        else:
            high_f = min(int(self.max_group_size), NF)

        FK = np.zeros((self._nb_of_groups, NF))
        for k in range(self._nb_of_groups):
            num_features = random_state.randint(1, high_f)
            rp = np.random.permutation(NF)
            FK[k, rp[0:num_features]] = 1

        components_ = np.zeros((NF, NF), dtype=Xz.dtype)

        n_samples = Xz.shape[0]

        for k in range(self._nb_of_groups):
            pos = np.nonzero(FK[k, :])[0]

            Xzk = Xz[:, pos]
            # TODO : subsample of class

            pca = PCA(n_components=len(pos), whiten=False, copy=True, random_state=self.random_state)

            if self.bootstrap:
                while True:
                    ii_to_keep = (
                        random_state.randn(n_samples) <= 0.63
                    )  # boostrap probability that an index is in a bootstrap sample (limit N -> inf)
                    index_to_keep = np.where(ii_to_keep)[0]
                    if len(index_to_keep) > 0:
                        # To prevent the (very unlickely) case where nothing is selected...)
                        break

                Xzk_bootstrap = Xzk[index_to_keep, :]
            else:
                Xzk_bootstrap = Xzk

            pca.fit(Xzk_bootstrap)

            rot = pca.components_.T
            assert rot.shape[0] == len(pos)
            if rot.shape[1] < len(pos):
                rot = np.hstack((rot, np.zeros((rot.shape[0], len(pos) - rot.shape[1]), dtype=rot.dtype)))

            assert rot.shape[0] == rot.shape[1]
            assert rot.shape[0] == len(pos)

            components_[pos.reshape(len(pos), 1), pos.reshape(1, len(pos))] = rot

        features_to_keep = np.any(components_ != 0, axis=0)
        self.components_ = components_[:, features_to_keep].astype(Xz.dtype)

        self._feature_names = ["RPCA_%d" % i for i in range(self.components_.shape[1])]

        return self

    def get_feature_names(self):
        return self._feature_names

    def transform(self, X):
        if self._scaler is None or self.components_ is None:
            raise NotFittedError("You should fit the model first")

        if get_type(X) != self._input_type:
            raise TypeError(
                "X should be a the same type as when fitted : %s, instead I got %s" % (self._input_type, type(X))
            )

        if X.shape[1] != self._nb_cols:
            raise ValueError(
                "X should have the same number of columns has when fitted (%d), instead I got %d"
                % (self._nb_cols, X.shape[1])
            )

        Xz = self._scaler.transform(X)
        Xzk_rot = np.dot(Xz, self.components_)

        return Xzk_rot


class GroupPCADecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """ PCA on random group of features followed by a Decision Tree

    See : GroupPCA and DecisionTreeClassifier
    """

    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        class_weight=None,
        presort=False,
        pca_bootstrap=False,
        pca_max_nb_groups=0.25,
        pca_max_group_size=0.05,
    ):

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort

        self.pca_bootstrap = pca_bootstrap
        self.pca_max_nb_groups = pca_max_nb_groups
        self.pca_max_group_size = pca_max_group_size

        self._tree = None
        self._group_pca = None

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):

        #        self._saved_X = X

        self.n_features_ = X.shape[1]

        # 1) create GroupPCA
        self._group_pca = GroupPCA(
            random_state=self.random_state,
            bootstrap=self.pca_bootstrap,
            max_nb_groups=self.pca_max_nb_groups,
            max_group_size=self.pca_max_group_size,
        )

        # 2) Create Tree
        self._tree = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            class_weight=self.class_weight,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            presort=self.presort,
        )

        # 3) Apply group PCA
        Xpca = self._group_pca.fit_transform(X, y)

        # 4) fit Tree
        self._tree.fit(Xpca, y, sample_weight=sample_weight, check_input=check_input, X_idx_sorted=None)

        return self

    def predict(self, X, check_input=True):
        if self._tree is None:
            raise NotFittedError("You should fit the model first")

        Xpca = self._group_pca.transform(X)
        return self._tree.predict(Xpca, check_input=check_input)

    def predict_proba(self, X, check_input=True):
        if self._tree is None:
            raise NotFittedError("You should fit the model first")

        Xpca = self._group_pca.transform(X)

        return self._tree.predict_proba(Xpca, check_input=check_input)

    def predict_log_proba(self, X, check_input=True):
        if self._tree is None:
            raise NotFittedError("You should fit the model first")

        Xpca = self._group_pca.transform(X)

        return self._tree.predict_proba(Xpca, check_input=check_input)

    def apply(self, X, check_input=True):
        if self._tree is None:
            raise NotFittedError("You should fit the model first")

        Xpca = self._group_pca.transform(X)

        return self._tree.apply(Xpca, check_input=check_input)

    def decision_path(self, X, check_input=True):
        Xpca = self._group_pca.transform(X)

        return self._tree.decision_path(Xpca, check_input=check_input)

    @property
    def tree_(self):
        return self._tree.tree_

    @property
    def classes_(self):
        return self._tree.classes_

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based " "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError(
                "Number of features of the model must "
                "match the input. Model n_features is %s and "
                "input n_features is %s " % (self.n_features_, n_features)
            )

        return X


class GroupPCADecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """ PCA on random group of features followed by a Decision Tree
    
    See : GroupPCA and DecisionTreeRegressor
    """

    def __init__(
        self,
        criterion="mse",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        presort=False,
        pca_bootstrap=False,
        pca_max_nb_groups=0.25,
        pca_max_group_size=0.05,
    ):

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.presort = presort

        self.pca_bootstrap = pca_bootstrap
        self.pca_max_nb_groups = pca_max_nb_groups
        self.pca_max_group_size = pca_max_group_size

        self._tree = None
        self._group_pca = None

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):

        self.n_features_ = X.shape[1]

        # 1) create GroupPCA
        self._group_pca = GroupPCA(
            random_state=self.random_state,
            bootstrap=self.pca_bootstrap,
            max_nb_groups=self.pca_max_nb_groups,
            max_group_size=self.pca_max_group_size,
        )
        # 2) Create Tree
        self._tree = DecisionTreeRegressor(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            presort=self.presort,
        )

        # 3) Apply group PCA
        Xpca = self._group_pca.fit_transform(X, y)

        # 4) fit Tree
        self._tree.fit(Xpca, y, sample_weight=sample_weight, check_input=check_input, X_idx_sorted=None)

        return self

    def predict(self, X, check_input=True):

        if self._tree is None:
            raise NotFittedError("You should fit the model first")

        Xpca = self._group_pca.transform(X)
        return self._tree.predict(Xpca, check_input=check_input)

    def apply(self, X, check_input=True):

        if self._tree is None:
            raise NotFittedError("You should fit the model first")

        Xpca = self._group_pca.transform(X)

        return self._tree.apply(Xpca, check_input=check_input)

    def decision_path(self, X, check_input=True):
        Xpca = self._group_pca.transform(X)

        return self._tree.decision_path(Xpca, check_input=check_input)

    @property
    def tree_(self):
        return self._tree.tree_

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based " "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError(
                "Number of features of the model must "
                "match the input. Model n_features is %s and "
                "input n_features is %s " % (self.n_features_, n_features)
            )

        return X


# In[]


class RandomRotationForestClassifier(ForestClassifier):
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
        pca_bootstrap=True,
        pca_max_nb_groups=0.25,
        pca_max_group_size=0.5,
    ):

        super(RandomRotationForestClassifier, self).__init__(
            base_estimator=GroupPCADecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
                "pca_bootstrap",
                "pca_max_nb_groups",
                "pca_max_group_size",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        self.pca_bootstrap = pca_bootstrap
        self.pca_max_nb_groups = pca_max_nb_groups
        self.pca_max_group_size = pca_max_group_size


class RandomRotationForestRegressor(ForestRegressor):
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
        pca_bootstrap=True,
        pca_max_nb_groups=0.25,
        pca_max_group_size=0.05,
    ):

        super(RandomRotationForestRegressor, self).__init__(
            base_estimator=GroupPCADecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "min_impurity_split",
                "random_state",
                "pca_bootstrap",
                "pca_max_nb_groups",
                "pca_max_group_size",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        self.pca_bootstrap = pca_bootstrap
        self.pca_max_nb_groups = pca_max_nb_groups
        self.pca_max_group_size = pca_max_group_size


# In[]
