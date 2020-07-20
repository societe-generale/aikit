# -*- coding: utf-8 -*-
"""
"""
import pytest

import numpy as np
import pandas as pd
import scipy.sparse as sps

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

from aikit.models import DBSCANWrapper, KMeansWrapper, AgglomerativeClusteringWrapper


def test_KMeans_fix_seed():
    Xtrain, cl = make_blobs(n_samples=1000, n_features=10, centers=5, random_state=123)
    first_label = []
    for i in range(10):
        kmeans = KMeans(n_clusters=5, random_state=123, n_init=1) # If n_init 
        kmeans.fit(Xtrain)
        first_label.append(kmeans.labels_[0])
    assert len(set(first_label)) == 1 # always the same ...


def test_KMeansWrapper():
    Xtrain, cl = make_blobs(n_samples=1000, n_features=10, centers=5, random_state=123)

    kmeans_wrapper = KMeansWrapper(n_clusters=5, random_state=123, n_init=1)
    kmeans_wrapper.fit(Xtrain)

    kmeans = KMeans(n_clusters=5, random_state=123)
    kmeans.fit(Xtrain)

    assert kmeans_wrapper.n_clusters == kmeans.n_clusters

    diff_cluster_centers = np.abs(kmeans_wrapper.cluster_centers_ - kmeans.cluster_centers_)
    
    assert np.array_equal(kmeans_wrapper.labels_, kmeans.labels_)
    assert np.sum(diff_cluster_centers) <= 10 ** (-10)

    # check that n_clusters is < n_features
    Xtrain, cl = make_blobs(n_samples=10, n_features=5, centers=5)

    kmeans_wrapper = KMeansWrapper(n_clusters=15, random_state=123, n_init=1)
    kmeans_wrapper.fit(Xtrain)

    assert kmeans_wrapper.n_clusters == 9


def test_AgglomerativeClusteringWrapper():
    Xtrain, cl = make_blobs(n_samples=1000, n_features=10, centers=5, random_state=123)

    agg_wrapper = AgglomerativeClusteringWrapper(n_clusters=5)
    agg_wrapper.fit(Xtrain)

    agg = AgglomerativeClustering(n_clusters=5)
    agg.fit(Xtrain)

    assert agg.n_clusters == agg_wrapper.n_clusters
    assert np.array_equal(agg.labels_, agg_wrapper.labels_)

    # check that n_clusters is < n_features
    Xtrain, cl = make_blobs(n_samples=10, n_features=5, centers=5)

    agg_wrapper = AgglomerativeClusteringWrapper(n_clusters=15)
    agg_wrapper.fit(Xtrain)

    assert agg_wrapper.n_clusters <= 10


def test_DBSCANWrapper():
    Xtrain, cl = make_blobs(n_samples=1000, n_features=10, centers=5, random_state=123)

    dbscan_wrapper = DBSCANWrapper(eps=0.5, scale_eps=False)
    dbscan_wrapper.fit(Xtrain)

    dbscan = DBSCAN(eps=0.5)
    dbscan.fit(Xtrain)

    assert np.array_equal(dbscan.labels_, dbscan_wrapper.labels_)

    Xtrain = np.mgrid[-1:1.2:0.2, -1:1.2:0.2].reshape(2, -1).T
    dbscan_wrapper = DBSCANWrapper(eps=0.5, scale_eps=True)
    dbscan_wrapper.fit(Xtrain)

    eps_to_find = 0.5 * (2.0 * np.sqrt(2.0) - 0.2) + 0.2

    assert np.abs(dbscan_wrapper.eps - eps_to_find) <= 10 ** (-5)
