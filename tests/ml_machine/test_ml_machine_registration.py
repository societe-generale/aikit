# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:25:29 2018

@author: Lionel Massoulard
"""
import numpy as np

from aikit.ml_machine.ml_machine_registration import LGBMClassifier_Model, CountVectorizer_TextEncoder
from aikit.ml_machine.ml_machine_registration import MODEL_REGISTER

from aikit.model_definition import DICO_NAME_KLASS


def test_LGBMClassifier_Model():
    hyper_gen = LGBMClassifier_Model.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(1000)]
    for hyper in all_hypers:
        if hyper["bagging_freq"] == 0:
            assert hyper["bagging_fraction"] == 1.0
            assert hyper["boosting_type"] != "rf"
        else:
            assert hyper["bagging_fraction"] < 1.0


def test_CountVectorizer_TextEncoder():
    hyper_gen = CountVectorizer_TextEncoder.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(1000)]
    for hyper in all_hypers:
        if hyper["analyzer"] == "word":
            assert hyper["ngram_range"] == 1


def verif_all():
    test_CountVectorizer_TextEncoder()
    test_LGBMClassifier_Model()


def test_hyper_init():
    np.random.seed(123)
    for model, hyper in MODEL_REGISTER.hyper_parameters.items():

        klass = DICO_NAME_KLASS[model[1]]
        klass(*hyper.get_rand())
