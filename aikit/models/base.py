# -*- coding: utf-8 -*-
"""
"""
import numpy as np

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances


class AgglomerativeClusteringWrapper(AgglomerativeClustering):
    """Wrapper around sklearn :class:`AgglomerativeClustering` with additional capabilities:

    * ensure that n_cluster is smaller than the number of samples and greater than 1
    """

    def __init__(
        self,
        n_clusters=2,
        affinity="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None
    ):
        super(AgglomerativeClusteringWrapper, self).__init__(
            n_clusters=n_clusters,
            affinity=affinity,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold
        )

    def fit(self, X, y=None):
        self.n_clusters = max(min(self.n_clusters, X.shape[0] - 1), 2)
        super(AgglomerativeClusteringWrapper, self).fit(X, y)
        return self


class KMeansWrapper(KMeans):
    """Wrapper around sklearn :class:`KMeans` with additional capabilities:

    * ensure that n_cluster is smaller than the number of samples and greater than 1
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=0.0001,
        precompute_distances="auto",
        verbose=0,
        random_state=None,
        copy_x=True,
        n_jobs=None,
        algorithm="auto",
    ):
        super(KMeansWrapper, self).__init__(
            n_clusters,
            init,
            n_init,
            max_iter,
            tol,
            precompute_distances,
            verbose,
            random_state,
            copy_x,
            n_jobs,
            algorithm,
        )

    def fit(self, X, y=None, sample_weight=None):
        self.n_clusters = max(min(self.n_clusters, X.shape[0] - 1), 2)
        super(KMeansWrapper, self).fit(X, y, sample_weight)
        return self


class DBSCANWrapper(DBSCAN):
    """Wrapper around sklearn :class:`DBSCAN` with additional capabilities:

    * transform the eps
    """

    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
        scale_eps=False,
    ):
        super(DBSCANWrapper, self).__init__(eps, min_samples, metric, metric_params, algorithm, leaf_size, p, n_jobs)
        self._scale_eps = scale_eps

    def fit(self, X, y=None, sample_weight=None):
        if self._scale_eps:
            self.eps = self.compute_eps(X)
        super(DBSCANWrapper, self).fit(X, y, sample_weight)
        return self

    def compute_eps(self, X):
        self._scale_eps = False
        if self.p:
            distances = pairwise_distances(X, metric=self.metric, n_jobs=self.n_jobs, p=self.p)
        else:
            distances = pairwise_distances(X, metric=self.metric, n_jobs=self.n_jobs)
        min_distance = np.min(distances[np.nonzero(distances)])
        max_distance = np.max(distances)

        if min_distance >= max_distance:
            return 1.0
        else:
            return min_distance + self.eps * (max_distance - min_distance)

    def get_params(self, deep=True):
        out = super(DBSCANWrapper, self).get_params(deep)
        out.pop("scale_eps", None)
        return out
