# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:03:19 2018

@author: Lionel Massoulard
"""

import pytest
import pandas as pd

from tests.helpers.testing_help import get_sample_df

from aikit.transformers.text import (
    CountVectorizerWrapper,
    TextDefaultProcessing,
    TextDigitAnonymizer,
    Word2VecVectorizer,
    Char2VecVectorizer,
    Word2Vec
)
from aikit.datasets.datasets import load_dataset


def test_TextDefaultProcessing():
    text = TextDefaultProcessing()

    examples = [
        ("process", "process"),
        ("process!", "process ."),
        ("text preprocessing, this is SO FUN###", "text preprocessing this is so fun"),
        ("something with digit 1234", "something with digit 1234"),
    ]

    for s, expected_result in examples:
        assert text.process_one_string(s) == expected_result


def test_CountVectorizerWrapper():

    df = get_sample_df(size=100, seed=123)

    vect = CountVectorizerWrapper(columns_to_use=["text_col"])
    vect.fit(df)

    cols = vect.get_feature_names()
    for c in cols:
        assert c.startswith("text_col__BAG")

    vect = CountVectorizerWrapper(columns_to_use=[2])
    vect.fit(df)

    cols = vect.get_feature_names()
    for c in cols:
        assert c.startswith("text_col__BAG")

    X = df.values
    vect = CountVectorizerWrapper(columns_to_use=[2])
    vect.fit(X)
    cols = vect.get_feature_names()
    for c in cols:
        assert c.startswith("2__BAG")


def test_CountVectorizerWrapper_on_Serie():

    df = get_sample_df(size=100, seed=123)

    X = df["text_col"]
    vect = CountVectorizerWrapper()

    Xres = vect.fit_transform(X)

    assert len(Xres.shape) == 2
    assert Xres.shape[0] == X.shape[0]
    assert Xres.shape[1] == len(vect.get_feature_names())

    Xres = vect.transform(X)
    assert len(Xres.shape) == 2
    assert Xres.shape[0] == X.shape[0]
    assert Xres.shape[1] == len(vect.get_feature_names())


def test_text_digit_anonymizer():
    transformer = TextDigitAnonymizer()
    df = pd.DataFrame(data=[["AAA", "A123"]])
    df_transformed = transformer.fit_transform(df)
    assert df_transformed is not None
    assert df_transformed.values.tolist() == [["AAA", "A###"]]

    try:
        transformer.fit_transform(pd.DataFrame(data=[[11354]]))
        pytest.fail("transform on non text data should fail with a TypeError")
    except TypeError:
        pass

@pytest.mark.skipif(Word2Vec is None, reason="gensim isn't installed")
def test_Word2VecVectorizer():
    df = get_sample_df(size=200, seed=123)

    ### default mode : 'drop' ##"
    vect = Word2VecVectorizer(columns_to_use=["text_col"], window=100)
    vect.fit(df)

    Xres = vect.transform(df)

    assert Xres.shape == (200, 100)
    assert not pd.isnull(Xres).any().any()
    assert vect.get_feature_names() == ["text_col__EMB__%d" % i for i in range(100)]
    assert list(Xres.columns) == vect.get_feature_names()

    ### keep mode ###
    vect = Word2VecVectorizer(columns_to_use=["text_col"], window=100, keep_other_columns="keep")
    vect.fit(df)

    Xres = vect.transform(df)

    assert Xres.shape == (200, 100 + df.shape[1])
    assert not pd.isnull(Xres).any().any()
    assert vect.get_feature_names() == list(df.columns) + ["text_col__EMB__%d" % i for i in range(100)]
    assert list(Xres.columns) == vect.get_feature_names()

    cols = [c for c in list(df.columns) if c in list(Xres.columns)]
    assert (Xres.loc[:, cols] == df.loc[:, cols]).all().all()

    ### delta mode ###
    vect = Word2VecVectorizer(columns_to_use=["text_col"], window=100, keep_other_columns="delta")
    vect.fit(df)

    Xres = vect.transform(df)

    assert Xres.shape == (200, 100 + df.shape[1] - 1)
    assert not pd.isnull(Xres).any().any()
    assert vect.get_feature_names() == [c for c in list(df.columns) if c != "text_col"] + [
        "text_col__EMB__%d" % i for i in range(100)
    ]
    assert list(Xres.columns) == vect.get_feature_names()

    cols = [c for c in list(df.columns) if c in list(Xres.columns)]
    assert (Xres.loc[:, cols] == df.loc[:, cols]).all().all()


@pytest.mark.skipif(Word2Vec is None, reason="gensim isn't installed")
def test_Char2VecVectorizer():

    Xtrain = load_dataset("titanic")[0]
    df1 = Xtrain.loc[0:600, :]
    df2 = Xtrain.loc[600:, :]

    enc_kwargs = {"columns_to_use": ["name", "ticket"]}
    enc_kwargs["text_preprocess"] = None
    enc_kwargs["same_embedding_all_columns"] = True
    enc_kwargs["size"] = 50
    enc_kwargs["window"] = 5
    enc_kwargs["random_state"] = 123

    vect = Char2VecVectorizer(**enc_kwargs)
    vect.fit(df1)
    X1 = vect.transform(df1)
    X2 = vect.transform(df2)

    assert X1.shape[1] == X2.shape[1]
    assert X1.shape[1] == 50 * 2
    assert X1.shape[0] == df1.shape[0]
    assert X2.shape[0] == df2.shape[0]

    vect = Char2VecVectorizer(**enc_kwargs)
    X2 = vect.fit_transform(df1)

    enc_kwargs = {"columns_to_use": ["name"]}
    enc_kwargs["text_preprocess"] = "nltk"
    enc_kwargs["same_embedding_all_columns"] = False
    enc_kwargs["size"] = 50
    enc_kwargs["window"] = 5
    enc_kwargs["random_state"] = 123

    vect = Char2VecVectorizer(**enc_kwargs)
    vect.fit(df1)
    X1 = vect.transform(df1)
    X2 = vect.transform(df2)
    assert X1.shape[1] == X2.shape[1]
    assert X1.shape[1] == 50 * 1
    assert X1.shape[0] == df1.shape[0]
    assert X2.shape[0] == df2.shape[0]
