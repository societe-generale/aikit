# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:24:50 2018

@author: Lionel Massoulard
"""


class ProblemType:
    """ enumeration of type of Machine Learning problem """

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"

    alls = (REGRESSION, CLASSIFICATION, CLUSTERING)


class VariableType:
    """ enumeration of type """

    TEXT = "TEXT"
    NUM = "NUM"
    CAT = "CAT"

    alls = (TEXT, NUM, CAT)


class StepCategory(object):
    """ enumeration of type of category """

    TextPreprocessing = "TextPreprocessing"
    TextEncoder = "TextEncoder"
    TextDimensionReduction = "TextDimensionReduction"

    CategoryEncoder = "CategoryEncoder"
    NumericalEncoder = "NumericalEncoder"

    MissingValueImputer = "MissingValueImputer"

    Scaling = "Scaling"

    DimensionReduction = "DimensionReduction"

    FeatureExtraction = "FeatureExtraction"
    FeatureSelection = "FeatureSelection"

    Model = "Model"

    TargetTransformer = "TargetTransformer"
    UnderOverSampler = "UnderOverSampler"
    Stacking = "Stacking"

    alls = (
        TextPreprocessing,
        TextEncoder,
        TextDimensionReduction,
        CategoryEncoder,
        NumericalEncoder,
        MissingValueImputer,
        Scaling,
        DimensionReduction,
        FeatureExtraction,
        FeatureSelection,
        Model,
        TargetTransformer,
        UnderOverSampler,
        Stacking,
    )

    @staticmethod
    def is_composition_step(step):
        return step in (
            StepCategory.TargetTransformer,
            StepCategory.UnderOverSampler,
            StepCategory.Stacking,
        )

    @staticmethod
    def get_type_of_variable(step):
        if step in (
                StepCategory.TextPreprocessing,
                StepCategory.TextEncoder,
                StepCategory.TextDimensionReduction,
        ):
            return VariableType.TEXT
        elif step in (StepCategory.CategoryEncoder, ):
            return VariableType.CAT
        elif step in (StepCategory.NumericalEncoder, ):
            return VariableType.NUM
        else:
            return None


class DataTypes:
    """ enumeration of type of data handled """

    DataFrame = "DataFrame"
    Serie = "Serie"
    SparseDataFrame = "SparseDataFrame"
    NumpyArray = "NumpyArray"
    SparseArray = "SparseArray"

    alls = (DataFrame, Serie, SparseDataFrame, NumpyArray, SparseArray)


class SpecialModels:
    """ enumeration of special models """

    GraphPipeline = "GraphPipeline"
    Pipeline = "Pipeline"
    FeatureUnion = "FeatureUnion"
    ColumnsSelector = "ColumnsSelector"

    alls = (GraphPipeline, Pipeline, FeatureUnion, ColumnsSelector)
