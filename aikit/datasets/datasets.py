# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 10:21:41 2018

@author: Lionel Massoulard
"""
import logging

import os.path
import tarfile
import tempfile
from contextlib import closing
from urllib.error import HTTPError, URLError

import pandas as pd
import numpy as np
import urllib
from urllib.parse import urlparse

from aikit.tools.helper_functions import load_json


class DatasetEnum:
    """
    Enumeration of public dataset names.
    Can be used to load dataset:
    >>> titanic_dataset = load_dataset(DatasetEnum.titanic)
    """
    titanic = "titanic"
    electricity = "electricity"
    housing = "housing"
    quora = "quora"
    abalone = "abalone"
    imdb = "imdb"
    pokemon = "pokemon"
    wikinews = "wikinews"
    school = "school"
    alls = (titanic, electricity, housing, quora, abalone, imdb, pokemon, school)


def get_dataset_config():
    """ return the folder of the datasets """
    config_file = os.path.join(os.path.split(__file__)[0], "config.json")
    if not os.path.exists(config_file):
        return None
    return load_json(config_file)


DATASET_CONFIG = get_dataset_config()


DATASET_PUBLIC_URLS = {
    'titanic': 'https://github.com/gfournier/aikit-datasets/releases/download/titanic-1.0.0/titanic.tar.gz'
}


def _load_public_path(url, cache_dir=None, cache_subdir='datasets'):
    """
    Load a public dataset from the specified URL. The data is loaded locally in the cache directory.

    Parameters:
    -----------
        url: string
            Dataset URL
        cache_dir: string, optional (default=None)
            Local cache directory, defaults to $AIKIT_HOME then ~/.aikit then $TMP/.aikit if None
        cache_subdir: string, optional (deault='datasets')
            Cache subdirectory

    Returns:
    --------
        path: string
            Path to the downloaded file
    """
    if cache_dir is None:
        if 'AIKIT_HOME' in os.environ:
            cache_dir = os.environ.get('AIKIT_HOME')
        else:
            cache_dir = os.path.join(os.path.expanduser('~'), '.aikit')
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join(tempfile.gettempdir(), '.aikit')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    fname = os.path.basename(urlparse(url).path)
    fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        error_msg = 'URL fetch failure on {} : {} -- {}'
        try:
            try:
                with closing(urllib.request.urlopen(url)) as response, open(fpath, 'wb') as fd:
                    fd.write(response.read())
            except HTTPError as e:
                raise Exception(error_msg.format(url, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(url, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    with tarfile.open(fpath, mode='r:gz') as tar:
        tar.extractall(os.path.join(datadir))

    return os.path.join(datadir, os.path.splitext(os.path.splitext(fname)[0])[0] + '.csv')


def find_path(name, cache_dir=None, cache_subdir='datasets'):
    """ find the path of a database """
    name = name.lower()

    if name in DATASET_PUBLIC_URLS:
        try:
            return _load_public_path(DATASET_PUBLIC_URLS[name], cache_dir=cache_dir, cache_subdir=cache_subdir)
        except:
            logging.getLogger("aikit.datasets").warning(
                "Failed to load public dataset from URL: {}, fallback to config.json file".format(
                    DATASET_PUBLIC_URLS[name]))

    if DATASET_CONFIG is None:
        config_file = os.path.join(os.path.split(__file__)[0], "config.json")
        raise ValueError("Please create a config.json file : %s" % str(config_file))

    config = DATASET_CONFIG.get(name, None)
    if config is None:
        config = DATASET_CONFIG["default"]
        path = os.path.join(config["path"], name, name + ".csv")
    else:
        path = config["path"]

    if not os.path.exists(path):
        raise ValueError("I couldn't find the database %s" % name)

    return path


def load_titanic(test_size=0.2, random_state=1, cache_dir=None, cache_subdir='datasets'):
    """ load titanic database """
    path = find_path(DatasetEnum.titanic, cache_dir=cache_dir, cache_subdir=cache_subdir)

    df = pd.read_csv(path, sep=",", na_values=["?"], keep_default_na=True)

    # Shuffle DF and compute train/test split
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    idx = int(len(df) * (1 - test_size))
    df_train = df.loc[:idx]
    df_test = df.loc[idx:]

    # Filter columns and build X, y
    y_train = df_train["survived"].values
    del df_train["survived"]
    y_test = df_test["survived"].values
    del df_test["survived"]
    infos = {}
    return df_train, y_train, df_test, y_test, infos


def load_imdb():
    """ load the imdb dataset """

    name = DatasetEnum.imdb
    path = find_path(name)

    df_train = pd.read_csv(os.path.join(path, "labeledTrainData.tsv"), sep="\t")
    y_train = df_train["sentiment"].values

    del df_train["id"]
    del df_train["sentiment"]

    df_test = pd.read_csv(os.path.join(path, "testData.tsv"), sep="\t")
    del df_test["id"]
    y_test = None

    # Rmk : unlabeled data is also present

    infos = {}
    return df_train, y_train, df_test, y_test, infos


def load_electricity():
    """ load electricity database """

    name = DatasetEnum.electricity

    path = find_path(name)

    df_train = pd.read_csv(os.path.join(path, "training_inputs.csv"), sep=";")
    y_train = pd.read_csv(os.path.join(path, "training_outputs.csv"), sep=";")

    assert (df_train["ID"] == y_train["ID"]).all()
    del df_train["ID"]
    y_train = y_train["TARGET"].values

    df_test = None
    y_test = None
    infos = {}

    return df_train, y_train, df_test, y_test, infos


def load_housing():
    """ load housing database """

    name = DatasetEnum.housing

    path = find_path(name)

    df_train = pd.read_csv(os.path.join(path, "train.csv"), sep=",")
    df_test = pd.read_csv(os.path.join(path, "test.csv"), sep=",")

    del df_train["Id"]
    del df_test["Id"]

    y_train = df_train["SalePrice"].values
    y_test = None  # df_test["SalePrice"].values

    del df_train["SalePrice"]

    infos = {}

    return df_train, y_train, df_test, y_test, infos


def load_quora():
    """ load quora database """

    name = DatasetEnum.quora

    path = find_path(name)

    df_train = pd.read_csv(os.path.join(path, "train.csv"), sep=",")
    y_train = df_train["is_duplicate"].values

    del df_train["id"]
    del df_train["qid1"]
    del df_train["qid2"]
    del df_train["is_duplicate"]

    df_test = None
    y_test = None
    infos = {}

    ii1 = df_train["question1"].isnull()
    ii2 = df_train["question2"].isnull()
    ii_to_keep = ~(ii1 | ii2)

    df_train = df_train.loc[ii_to_keep, :]
    df_train.index = range(len(df_train))

    y_train = y_train[ii_to_keep]

    return df_train, y_train, df_test, y_test, infos


def load_abalone():
    """ load abalone dataset """

    name = DatasetEnum.abalone

    path = find_path(name)

    df_train = pd.read_csv(os.path.join(path, "train.csv"), sep=",")
    y_train = df_train["rings"].values

    del df_train["rings"]

    df_test = None
    y_test = None
    infos = {}

    return df_train, y_train, df_test, y_test, infos


def load_pokemon():
    """ pokemon database """
    name = DatasetEnum.pokemon

    df_test = None
    y_train = None
    y_test = None
    # This is more of a clustering dataset ... no special targetinfos={}
    path = find_path(name)

    df_train = pd.read_csv(os.path.join(path, "Pokemon.csv"), sep=",")
    del df_train["#"]

    infos = {}
    return df_train, y_train, df_test, y_test, infos


def load_wikinews():
    """ wikipedia news dataset """
    name = DatasetEnum.wikinews

    df_test = None
    y_train = None
    y_test = None

    path = find_path(name)

    df_train = pd.read_csv(os.path.join(path, "wiki_stories_20150102_20181211.csv"), sep=",")

    del df_train["id"]

    infos = {}

    return df_train, y_train, df_test, y_test, infos


def load_school():
    """ wikipedia news dataset """
    name = DatasetEnum.school

    df_test = None
    y_train = None
    y_test = None

    path = find_path(name)

    df = pd.read_csv(os.path.join(path, "school_results.csv"), sep=",")
    to_drop = ["G1", "G2", "G3"]
    df["results"] = np.mean(df.loc[:, to_drop], axis=1)
    to_drop.append("results")
    df_train = df.copy().drop(to_drop, axis=1)
    y_train = df["results"]

    infos = {}

    return df_train, y_train, df_test, y_test, infos


def load_dataset(name, cache_dir=None, cache_subdir='datasets'):
    """ loading datasets """

    if name == DatasetEnum.titanic:
        res = load_titanic(cache_dir=cache_dir, cache_subdir=cache_subdir)

    elif name == DatasetEnum.electricity:
        res = load_electricity()

    elif name == DatasetEnum.housing:
        res = load_housing()

    elif name == DatasetEnum.quora:
        res = load_quora()

    elif name == DatasetEnum.abalone:
        res = load_abalone()

    elif name == DatasetEnum.imdb:
        res = load_imdb()

    elif name == DatasetEnum.pokemon:
        res = load_pokemon()

    elif name == DatasetEnum.wikinews:
        res = load_wikinews()

    elif name == DatasetEnum.school:
        res = load_school()

    else:
        raise ValueError("I don't know this database %s" % name)

    return res
