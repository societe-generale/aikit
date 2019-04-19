# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:55:08 2018

@author: Lionel Massoulard
"""
import pytest

import numpy as np
import pandas as pd
import scipy.sparse as sps

from sklearn.datasets import make_blobs
from sklearn.linear_model import Ridge
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer

from tests.helpers.testing_help import get_sample_data, get_sample_df

from aikit.tools.helper_functions import diff
from aikit.tools.data_structure_helper import get_type
from aikit.enums import DataTypes
from aikit.cross_validation import create_scoring

from aikit.transformers.base import (
    TruncatedSVDWrapper,
    _KMeansTransformer,
    KMeansTransformer,
    _NumImputer,
    NumImputer,
    BoxCoxTargetTransformer,
    _CdfScaler,
    CdfScaler,
    PCAWrapper,
)
from aikit.transformers.base import _index_with_number, PassThrough, FeaturesSelectorClassifier

# In[]


def test_TruncatedSVDWrapper():

    df = get_sample_df(100, seed=123)
    cols = []
    for j in range(10):
        cols.append("num_col_%d" % j)
        df["num_col_%d" % j] = np.random.randn(df.shape[0])

    # 1) regular case : drop other columns
    svd = TruncatedSVDWrapper(n_components=5, columns_to_use=cols)
    res1 = svd.fit_transform(df)

    assert res1.shape == (100, 5)
    assert get_type(res1) == DataTypes.DataFrame
    assert list(res1.columns) == ["SVD__%d" % j for j in range(5)]
    assert not res1.isnull().any().any()
    assert svd.get_feature_names() == list(res1.columns)

    # 2) we keep the original columns as well
    svd = TruncatedSVDWrapper(n_components=5, columns_to_use=cols, keep_other_columns="keep")
    res2 = svd.fit_transform(df)

    assert res2.shape == (100, 5 + df.shape[1])

    assert get_type(res2) == DataTypes.DataFrame
    assert list(res2.columns) == list(df.columns) + ["SVD__%d" % j for j in range(5)]
    assert svd.get_feature_names() == list(df.columns) + ["SVD__%d" % j for j in range(5)]
    assert not res2.isnull().any().any()
    assert (res2.loc[:, list(df.columns)] == df).all().all()

    # 3) we keep only untouch columns
    svd = TruncatedSVDWrapper(n_components=5, columns_to_use=cols, keep_other_columns="delta")
    res3 = svd.fit_transform(df)
    assert res3.shape == (100, 3 + 5)
    assert list(res3.columns) == ["float_col", "int_col", "text_col"] + ["SVD__%d" % j for j in range(5)]
    assert svd.get_feature_names() == ["float_col", "int_col", "text_col"] + ["SVD__%d" % j for j in range(5)]
    assert (
        (res3.loc[:, ["float_col", "int_col", "text_col"]] == df.loc[:, ["float_col", "int_col", "text_col"]])
        .all()
        .all()
    )

    ###################################
    ###  same thing but with regex  ###
    ###################################

    # 1) Regular case : 'drop' other columns
    svd = TruncatedSVDWrapper(n_components=5, columns_to_use=["num_col_"], regex_match=True)
    res1 = svd.fit_transform(df)
    assert res1.shape == (100, 5)
    assert get_type(res1) == DataTypes.DataFrame
    assert list(res1.columns) == ["SVD__%d" % j for j in range(5)]
    assert not res1.isnull().any().any()
    assert svd.get_feature_names() == list(res1.columns)

    # 2) Keep original columns
    svd = TruncatedSVDWrapper(n_components=5, columns_to_use=["num_col_"], keep_other_columns="keep", regex_match=True)
    res2 = svd.fit_transform(df)

    assert res2.shape == (100, 5 + df.shape[1])

    assert get_type(res2) == DataTypes.DataFrame
    assert list(res2.columns) == list(df.columns) + ["SVD__%d" % j for j in range(5)]
    assert svd.get_feature_names() == list(df.columns) + ["SVD__%d" % j for j in range(5)]
    assert not res2.isnull().any().any()
    assert (res2.loc[:, list(df.columns)] == df).all().all()

    # 3) Keep only the un-touch column
    svd = TruncatedSVDWrapper(n_components=5, columns_to_use=["num_col_"], keep_other_columns="delta", regex_match=True)
    res3 = svd.fit_transform(df)
    assert res3.shape == (100, 3 + 5)
    assert list(res3.columns) == ["float_col", "int_col", "text_col"] + ["SVD__%d" % j for j in range(5)]
    assert svd.get_feature_names() == ["float_col", "int_col", "text_col"] + ["SVD__%d" % j for j in range(5)]
    assert (
        (res3.loc[:, ["float_col", "int_col", "text_col"]] == df.loc[:, ["float_col", "int_col", "text_col"]])
        .all()
        .all()
    )

    # Delta with numpy ###
    xx = df.values
    columns_to_use = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    svd = TruncatedSVDWrapper(n_components=5, columns_to_use=columns_to_use, keep_other_columns="delta")
    res4 = svd.fit_transform(xx)
    assert list(res4.columns) == [0, 1, 2] + ["SVD__%d" % i for i in range(5)]
    assert svd.get_feature_names() == [0, 1, 2] + ["SVD__%d" % i for i in range(5)]

    input_features = ["COL_%d" % i for i in range(xx.shape[1])]
    assert svd.get_feature_names(input_features) == ["COL_0", "COL_1", "COL_2"] + ["SVD__%d" % i for i in range(5)]

    # Keep
    svd = TruncatedSVDWrapper(n_components=5, columns_to_use=columns_to_use, keep_other_columns="keep")
    res2 = svd.fit_transform(xx)
    assert list(res2.columns) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + ["SVD__%d" % i for i in range(5)]
    assert svd.get_feature_names() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + ["SVD__%d" % i for i in range(5)]
    assert svd.get_feature_names(input_features) == input_features + ["SVD__%d" % i for i in range(5)]


def test_FeaturesSelectorClassifier_get_feature_names():

    vect = CountVectorizer(analyzer="char", ngram_range=(1, 3))

    df = get_sample_df(100, seed=123)
    xx = vect.fit_transform(df["text_col"])
    y = 1 * (np.random.rand(xx.shape[0]) > 0.5)

    sel = FeaturesSelectorClassifier(n_components=10)
    sel.fit_transform(xx, y)

    ff0 = vect.get_feature_names()
    ff1 = sel.get_feature_names()

    assert len(diff(ff1, list(range(xx.shape[1])))) == 0

    ff2 = sel.get_feature_names(input_features=ff0)

    assert len(ff1) == len(ff2)

    for f1, f2 in zip(ff1, ff2):
        assert ff0[f1] == f2


def test_KMeansTransformer():

    Xtrain, cl = make_blobs(n_samples=1000, n_features=10, centers=5)

    #    plt.plot(Xtrain[:,0],Xtrain[:,1],".")

    kmeans = _KMeansTransformer(n_clusters=5, result_type="probability")
    kmeans.fit(Xtrain)

    Xres = kmeans.transform(Xtrain)

    X = np.array([[0] * 10])
    kmeans.transform(X)

    kmeans2 = _KMeansTransformer(n_clusters=5, result_type="probability", temperature=0.1)
    Xres2 = kmeans2.fit_transform(Xtrain)

    kmeans3 = _KMeansTransformer(n_clusters=5, result_type="probability", temperature=0.01)
    Xres3 = kmeans3.fit_transform(Xtrain)

    p = Xres.max(axis=1)
    p2 = Xres2.max(axis=1)
    p3 = Xres3.max(axis=1)

    # plt.plot(p,p2,".")
    # plt.plot(p,p3,".")
    # plt.plot(p2,p3,".")

    assert p.mean() > p2.mean()
    assert p2.mean() > p3.mean()

    assert Xres.shape == (1000, 5)
    assert Xres.min() >= 0
    assert Xres.max() <= 1
    assert pd.isnull(Xres).sum() == 0

    assert np.max(np.abs(Xres.sum(axis=1) - 1)) <= 10 ** (-10)

    for result_type in ("distance", "log_distance", "inv_distance", "probability"):
        kmeans = _KMeansTransformer(n_clusters=5, result_type=result_type, random_state=123)
        kmeans.fit(Xtrain)

        Xres = kmeans.transform(Xtrain)
        Xres2 = kmeans.transform(X)

        assert Xres.shape == (1000, 5)
        assert Xres.min() >= 0
        assert pd.isnull(Xres).sum() == 0

    ### Same thing but with Wrapper
    kmeans = KMeansTransformer(n_clusters=5, result_type="probability")
    kmeans.fit(Xtrain)

    Xres = kmeans.transform(Xtrain)

    kmeans2 = KMeansTransformer(n_clusters=5, result_type="probability", temperature=0.1)
    Xres2 = kmeans2.fit_transform(Xtrain)

    kmeans3 = KMeansTransformer(n_clusters=5, result_type="probability", temperature=0.01)
    Xres3 = kmeans3.fit_transform(Xtrain)

    p = Xres.max(axis=1)
    p2 = Xres2.max(axis=1)
    p3 = Xres3.max(axis=1)

    # plt.plot(p,p2,".")
    # plt.plot(p,p3,".")

    assert p.mean() > p2.mean()
    assert p2.mean() > p3.mean()

    assert Xres.shape == (1000, 5)
    assert Xres.min().min() >= 0
    assert Xres.max().max() <= 1
    assert Xres.isnull().sum().sum() == 0

    assert np.max(np.abs(Xres.sum(axis=1) - 1)) <= 10 ** (-10)

    for result_type in ("distance", "log_distance", "inv_distance", "probability"):
        kmeans = KMeansTransformer(n_clusters=5, result_type=result_type, random_state=123)
        kmeans.fit(Xtrain)

        Xres = kmeans.transform(Xtrain)
        Xres2 = kmeans.transform(X)

        assert Xres.shape == (1000, 5)
        assert Xres.min().min() >= 0
        assert Xres.isnull().sum().sum() == 0


def test__index_with_number():
    x = np.random.randn(10)
    assert _index_with_number(x).all()

    res = np.array([True] * 10)

    x2 = x.copy().astype(np.object)
    res[[0, 1, 3]] = False
    x2[[0, 1, 3]] = "string"

    assert (_index_with_number(x2) == res).all()


def test__NumImputer():

    xx, xxd, xxs = get_sample_data(add_na=True)
    xxd.index = np.array([0, 1, 2, 3, 4, 10, 11, 12, 12, 14])

    # DataFrame entry
    for inp in (_NumImputer(), NumImputer(), _NumImputer(add_is_null=False), NumImputer(add_is_null=False)):
        xx_out = inp.fit_transform(xxd)
        assert (xx_out.index == xxd.index).all()
        assert pd.isnull(xxd.loc[0, "col1"])  # Verify that it is still null
        assert xx_out.isnull().sum().sum() == 0
        assert xx_out["col1"][0] == xxd.loc[~xxd["col1"].isnull(), "col1"].mean()

        assert xx_out.shape[0] == xx.shape[0]
        assert get_type(xx_out) == get_type(xxd)

        if inp.add_is_null:
            assert inp.get_feature_names() == ["col0", "col1", "col2", "col3", "col4", "col5", "col6", "col1_isnull"]
            assert xx_out.shape[1] == 1 + xxd.shape[1]
            assert xx_out["col1_isnull"].iloc[0] == 1
            assert xx_out["col1_isnull"].iloc[5] == 1
            assert (xx_out["col1_isnull"].iloc[np.array([1, 2, 3, 4, 6, 7, 8, 9])] == 0).all()

        else:
            assert xx_out.shape[1] == xxd.shape[1]
            assert inp.get_feature_names() == ["col0", "col1", "col2", "col3", "col4", "col5", "col6"]

        inp = _NumImputer(add_is_null=False, allow_unseen_null=False)
        inp.fit(xxd)
        xxd2 = xxd.copy()
        xxd2.iloc[0, 3] = np.nan
        try:
            inp.transform(xxd2)
            raise AssertionError("Model should have fail its transformation")
        except ValueError:
            pass

    input_features = ["COL_%d" % i for i in range(xx.shape[1])]
    # Numpy array
    for inp in (_NumImputer(), NumImputer()):
        xx_out = inp.fit_transform(xx)
        assert pd.isnull(xx[0, 1])
        assert pd.isnull(xx_out).sum() == 0
        assert xx_out.shape[1] == 1 + xx.shape[1]
        assert xx_out.shape[0] == xx.shape[0]
        assert get_type(xx_out) == get_type(xx)
        assert inp.get_feature_names() == ["0", "1", "2", "3", "4", "5", "6", "1_isnull"]
        assert inp.get_feature_names(input_features) == input_features + ["COL_1_isnull"]
        assert xx_out[0, 7] == 1
        assert xx_out[5, 7] == 1
        assert (xx_out[np.array([1, 2, 3, 4, 6, 7, 8, 9]), 7] == 0).all()

    # Sparse Array
    for inp in (_NumImputer(), NumImputer()):
        for f in (sps.coo_matrix, sps.csc_matrix, sps.csr_matrix):
            xxsf = f(xxs.copy())
            xx_out = inp.fit_transform(xxsf)
            assert pd.isnull(xxs[0, 1])
            assert pd.isnull(xx_out.todense()).sum() == 0
            assert get_type(xx_out) == get_type(xxs)
            assert xx_out.shape[1] == 1 + xxs.shape[1]
            assert xx_out.shape[0] == xx.shape[0]
            assert inp.get_feature_names() == ["0", "1", "2", "3", "4", "5", "6", "1_isnull"]
            assert inp.get_feature_names(input_features) == input_features + ["COL_1_isnull"]
            assert xx_out.todense()[0, 7] == 1
            assert xx_out.todense()[0, 7] == 1
            assert (xx_out.todense()[np.array([1, 2, 3, 4, 6, 7, 8, 9]), 7] == 0).all()

    xx, xxd, xxs = get_sample_data(add_na=False)
    xxd.index = np.array([0, 1, 2, 3, 4, 10, 11, 12, 12, 14])

    # DataFrame entry
    for inp in (_NumImputer(), NumImputer()):
        xx_out = inp.fit_transform(xxd)
        assert (xx_out.index == xxd.index).all()
        assert xx_out.isnull().sum().sum() == 0
        assert xx_out.shape[1] == xxd.shape[1]
        assert xx_out.shape[0] == xx.shape[0]
        assert get_type(xx_out) == get_type(xxd)
        assert inp.get_feature_names() == ["col0", "col1", "col2", "col3", "col4", "col5", "col6"]

    # Numpy array
    for inp in (_NumImputer(), NumImputer()):
        xx_out = inp.fit_transform(xx)
        assert pd.isnull(xx_out).sum() == 0
        assert xx_out.shape[1] == xx.shape[1]
        assert xx_out.shape[0] == xx.shape[0]
        assert get_type(xx_out) == get_type(xx)
        assert inp.get_feature_names() == ["0", "1", "2", "3", "4", "5", "6"]
        assert inp.get_feature_names(input_features=input_features) == input_features

    # Sparse Array
    for inp in (_NumImputer(), NumImputer()):
        for f in (sps.coo_matrix, sps.csc_matrix, sps.csr_matrix):
            xxs_f = f(xxs.copy())
            xx_out = inp.fit_transform(xxs_f)
            assert pd.isnull(xx_out.todense()).sum() == 0
            assert get_type(xx_out) == get_type(xxs)
            assert xx_out.shape[1] == xxs.shape[1]
            assert xx_out.shape[0] == xx.shape[0]
            assert inp.get_feature_names() == ["0", "1", "2", "3", "4", "5", "6"]
            assert inp.get_feature_names(input_features=input_features) == input_features


def test_NumImputer_mixtype():
    df = get_sample_df(100, seed=123)
    df.loc[[2, 10, 50], "float_col"] = ["string", "string", "string"]

    imp = _NumImputer()

    Xenc = imp.fit_transform(df)

    assert _index_with_number(Xenc["float_col"]).all()
    assert not (Xenc.dtypes == "O").any()


def test_BoxCoxTargetTransformer_target_transform():

    for ll in (0, 0.1, 0.5, 2):

        bb = BoxCoxTargetTransformer(Ridge(), ll=ll)

        assert not is_classifier(bb)
        assert is_regressor(bb)

        y = np.arange(-100, 100, step=0.1)

        my = bb.target_transform(y)
        ymy = bb.target_inverse_transform(my)
        mymy = bb.target_transform(ymy)

        #        plt.subplot(211)
        #        plt.plot(y,my)
        #        plt.subplot(212)
        #        plt.plot(my,ymy)

        assert not pd.Series(my).isnull().any()
        assert not pd.Series(ymy).isnull().any()
        assert np.max(np.abs(y - ymy)) <= 10 ** (-10)
        assert np.max(np.abs(my - mymy)) <= 10 ** (-10)


def test_BoxCoxTargetTransformer():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.exp(np.random.randn(100))

    X2 = np.random.randn(100, 10) * 2

    for ll in (0, 0.1, 0.5, 2):

        bb = BoxCoxTargetTransformer(Ridge(), ll=ll)

        bb.fit(X, y)

        yhat = bb.predict(X)
        yhat2 = bb.predict(X2)

        assert yhat.ndim == 1
        assert yhat.shape[0] == y.shape[0]

        assert yhat2.ndim == 1
        assert yhat2.shape[0] == y.shape[0]


def test_approx_cross_validation_BoxCoxTargetTransformer():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.exp(np.random.randn(100))

    for ll in (0, 0.1, 0.5, 2):

        # Scorer entered as a string #

        bb = BoxCoxTargetTransformer(Ridge(), ll=ll)
        cv_res1, yhat1 = bb.approx_cross_validation(
            X, y, scoring=["neg_mean_squared_error"], cv=10, return_predict=True
        )

        assert isinstance(cv_res1, pd.DataFrame)
        assert cv_res1.shape[0] == 10
        assert "test_neg_mean_squared_error" in cv_res1
        assert "train_neg_mean_squared_error" in cv_res1

        assert yhat1.ndim == 1
        assert yhat1.shape[0] == y.shape[0]

        with pytest.raises(NotFittedError):
            bb.predict(X)

        with pytest.raises(NotFittedError):
            bb.model.predict(X)

        #########################################
        ###  Scorer entered as a dictionnary  ###
        #########################################
        scoring = create_scoring(Ridge(), ["neg_mean_squared_error"])
        cv_res2, yhat2 = bb.approx_cross_validation(X, y, scoring=scoring, cv=10, return_predict=True)

        assert isinstance(cv_res2, pd.DataFrame)
        assert cv_res2.shape[0] == 10
        assert "test_neg_mean_squared_error" in cv_res2
        assert "train_neg_mean_squared_error" in cv_res2

        assert yhat2.ndim == 1
        assert yhat2.shape[0] == y.shape[0]

        with pytest.raises(NotFittedError):
            bb.predict(X)

        with pytest.raises(NotFittedError):
            bb.model.predict(X)

        assert np.abs(cv_res2["test_neg_mean_squared_error"] - cv_res1["test_neg_mean_squared_error"]).max() <= 10 ** (
            -5
        )
        assert np.abs(
            cv_res2["train_neg_mean_squared_error"] - cv_res1["train_neg_mean_squared_error"]
        ).max() <= 10 ** (-5)

        assert np.max(np.abs(yhat2 - yhat1)) <= 10 ** (-5)


@pytest.mark.longtest
def test_CdfScaler():
    np.random.seed(123)

    # Array
    X = np.exp(np.random.randn(1000, 10))
    Xc = X.copy()

    # DataFrame
    dfX = pd.DataFrame(X, columns=["col_%d" % j for j in range(X.shape[1])])
    dfXc = dfX.copy()

    # Sparse Array
    Xsp = np.random.randn(1000, 10)
    Xsp[np.random.randn(1000, 10) <= 1] = 0
    Xsp = sps.csc_matrix(Xsp)

    Xspc = Xsp.copy()

    for klass in (_CdfScaler, CdfScaler):
        for output_distribution in ("uniform", "normal"):
            for distribution in ("normal", "auto-kernel", "auto-param", "auto-nonparam", "kernel", "none", "rank"):

                # Array
                scaler = klass(distribution=distribution, output_distribution=output_distribution)
                scaler.fit(X)

                Xs = scaler.transform(X)
                assert Xs.shape == X.shape

                if distribution != "none" and output_distribution == "uniform":
                    assert Xs.min() >= 0
                    assert Xs.max() <= 1

                elif distribution == "none":
                    assert (Xs == X).all()

                assert (X == Xc).all()  # verify that X didn't change
                assert not pd.isnull(X).any()

                # DataFrame
                scaler = klass(distribution=distribution, output_distribution=output_distribution)
                scaler.fit(dfX)

                dfXs = scaler.transform(dfX)
                assert dfXs.shape == dfX.shape
                assert list(dfXs.columns) == list(dfX.columns)

                if distribution != "none" and output_distribution == "uniform":
                    assert dfXs.min().min() >= 0
                    assert dfXs.max().max() <= 1
                else:
                    (dfXs == dfX).all().all()

                assert (dfXc == dfX).all().all()
                assert not dfXs.isnull().any().any()

                # Sparse Array
                scaler = klass(distribution=distribution, output_distribution=output_distribution)
                scaler.fit(Xsp)
                Xsps = scaler.transform(Xsp)
                assert isinstance(Xsps, sps.csc_matrix)
                assert (Xsps.indices == Xsp.indices).all()
                assert (Xsps.indptr == Xsp.indptr).all()
                if distribution != "none" and output_distribution == "uniform":
                    assert Xsps.data.min() >= 0
                    assert Xsps.data.max() <= 1

                if distribution != "none":
                    assert (Xsps.data != Xsp.data).any()

                assert not (Xspc != Xsp).todense().any()  # X didn't change in the process

    dfX = pd.DataFrame(
        {
            "A": np.random.randn(1000),
            "B": np.exp(np.random.randn(1000)),
            "C": np.random.rand(1000),
            "D": np.random.randint(0, 2, size=1000),
        }
    ).loc[:, ("A", "B", "C", "D")]

    dfXc = dfX.copy()

    scaler = CdfScaler(distribution="auto-param")
    scaler.fit(dfX)
    dfXs = scaler.transform(dfX)

    assert (dfXs["D"] == dfXc["D"]).all()
    assert dfXs.min().min() >= 0
    assert dfXs.max().max() <= 1

    assert scaler._model.distributions == ["normal", "gamma", "beta", "none"]


def test_PassThrough():

    df = get_sample_df(100, seed=123)
    pt = PassThrough()

    pt.fit(df)

    df2 = pt.transform(df)

    assert df2.shape == df.shape
    assert (df2 == df).all().all()
    assert id(df) == id(df2)

    assert pt.get_feature_names() == list(df.columns)

    with pytest.raises(ValueError):
        pt.transform(df.values)

    with pytest.raises(ValueError):
        pt.transform(df.iloc[:, [0, 1]])

    X = np.random.randn(20, 5)
    input_features = ["COL_%d" % i for i in range(5)]
    pt = PassThrough()
    pt.fit(X)

    X2 = pt.transform(X)

    assert X.shape == X2.shape  # same shape
    assert (X == X2).all()  # same value
    assert id(X) == id(X2)  # no copy

    assert pt.get_feature_names() == [0, 1, 2, 3, 4]
    assert pt.get_feature_names(input_features=input_features) == ["COL_0", "COL_1", "COL_2", "COL_3", "COL_4"]


def test_PCAWrapper():
    df = get_sample_df(100, seed=123)
    cols = []
    for j in range(10):
        cols.append("num_col_%d" % j)
        df["num_col_%d" % j] = np.random.randn(df.shape[0])

    # 0) n_components > n_features
    pca = PCAWrapper(n_components=15, columns_to_use=cols)
    res0 = pca.fit_transform(df)

    assert res0.shape == (100, len(cols) - 1)
    assert get_type(res0) == DataTypes.DataFrame
    assert list(res0.columns) == ["PCA__%d" % j for j in range(len(cols) - 1)]
    assert not res0.isnull().any().any()
    assert pca.get_feature_names() == list(res0.columns)

    # 1) regular case : drop other columns
    pca = PCAWrapper(n_components=5, columns_to_use=cols)
    res1 = pca.fit_transform(df)

    assert res1.shape == (100, 5)
    assert get_type(res1) == DataTypes.DataFrame
    assert list(res1.columns) == ["PCA__%d" % j for j in range(5)]
    assert not res1.isnull().any().any()
    assert pca.get_feature_names() == list(res1.columns)

    # 2) we keep the original columns as well
    pca = PCAWrapper(n_components=5, columns_to_use=cols, keep_other_columns="keep")
    res2 = pca.fit_transform(df)

    assert res2.shape == (100, 5 + df.shape[1])

    assert get_type(res2) == DataTypes.DataFrame
    assert list(res2.columns) == list(df.columns) + ["PCA__%d" % j for j in range(5)]
    assert pca.get_feature_names() == list(df.columns) + ["PCA__%d" % j for j in range(5)]
    assert not res2.isnull().any().any()
    assert (res2.loc[:, list(df.columns)] == df).all().all()

    # 3) Keep only the un-touch column
    pca = PCAWrapper(n_components=5, columns_to_use=["num_col_"], keep_other_columns="delta", regex_match=True)
    res3 = pca.fit_transform(df)
    assert res3.shape == (100, 3 + 5)
    assert list(res3.columns) == ["float_col", "int_col", "text_col"] + ["PCA__%d" % j for j in range(5)]
    assert pca.get_feature_names() == ["float_col", "int_col", "text_col"] + ["PCA__%d" % j for j in range(5)]
    assert (
        (res3.loc[:, ["float_col", "int_col", "text_col"]] == df.loc[:, ["float_col", "int_col", "text_col"]])
        .all()
        .all()
    )

    # Delta with numpy ###
    xx = df.values
    columns_to_use = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    pca = PCAWrapper(n_components=5, columns_to_use=columns_to_use, keep_other_columns="delta")
    res4 = pca.fit_transform(xx)
    assert list(res4.columns) == [0, 1, 2] + ["PCA__%d" % i for i in range(5)]
    assert pca.get_feature_names() == [0, 1, 2] + ["PCA__%d" % i for i in range(5)]

    input_features = ["COL_%d" % i for i in range(xx.shape[1])]
    assert pca.get_feature_names(input_features) == ["COL_0", "COL_1", "COL_2"] + ["PCA__%d" % i for i in range(5)]

    # Keep
    pca = PCAWrapper(n_components=5, columns_to_use=columns_to_use, keep_other_columns="keep")
    res2 = pca.fit_transform(xx)
    assert list(res2.columns) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + ["PCA__%d" % i for i in range(5)]
    assert pca.get_feature_names() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + ["PCA__%d" % i for i in range(5)]
    assert pca.get_feature_names(input_features) == input_features + ["PCA__%d" % i for i in range(5)]


def verif_all():
    test__NumImputer()
    test_BoxCoxTargetTransformer()
    test_KMeansTransformer()
    test__index_with_number()
    test_NumImputer_mixtype()
    test_FeaturesSelectorClassifier_get_feature_names()
    test_PassThrough()
