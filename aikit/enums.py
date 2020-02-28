# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:24:50 2018

@author: Lionel Massoulard
"""


# In[]
class TypeOfProblem:
    """ enumeration of type of Machine Learning problem """

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"

    alls = (REGRESSION, CLASSIFICATION, CLUSTERING)


# In[]


class TypeOfVariables:
    """ enumeration of type """

    TEXT = "TEXT"
    NUM = "NUM"
    CAT = "CAT"

    alls = (TEXT, NUM, CAT)


# In[]


class StepCategories(object):
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
        return step in (StepCategories.TargetTransformer, StepCategories.UnderOverSampler, StepCategories.Stacking)

    @staticmethod
    def get_type_of_variable(step):

        if step in (
            StepCategories.TextPreprocessing,
            StepCategories.TextEncoder,
            StepCategories.TextDimensionReduction,
        ):
            return TypeOfVariables.TEXT

        elif step in (StepCategories.CategoryEncoder,):
            return TypeOfVariables.CAT

        elif step in (StepCategories.NumericalEncoder,):
            return TypeOfVariables.NUM

        else:
            return None


# In[]


class DataTypes:
    """ enumeration of type of data handled """

    DataFrame = "DataFrame"
    Serie = "Serie"
    SparseDataFrame = "SparseDataFrame"
    NumpyArray = "NumpyArray"
    SparseArray = "SparseArray"

    alls = (DataFrame, Serie, SparseDataFrame, NumpyArray, SparseArray)


# In[]


class SpecialModels:
    """ enumeration of special models """

    GraphPipeline = "GraphPipeline"
    Pipeline = "Pipeline"
    FeatureUnion = "FeatureUnion"
    ColumnsSelector = "ColumnsSelector"

    alls = (GraphPipeline, Pipeline, FeatureUnion, ColumnsSelector)


