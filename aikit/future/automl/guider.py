# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:35:20 2018

@author: Lionel Massoulard
"""

import logging
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from scipy.stats import norm, rankdata
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from aikit.future.util.list import diff, intersect
from aikit.pipeline import GraphPipeline
from aikit.transformers import NumericalEncoder, NumImputer

logger = logging.getLogger(__name__)


def transform_quantile(array):
    """ transform an array into its rank , observation spaced between 1/2N and 1 - 1/2N every '1/N' """
    # nn = np.zeros(len(xx))
    # oo = np.argsort(xx)
    # nn[oo] = np.arange(len(xx)) / len(xx) + 1 / (2 * len(xx))
    # return nn
    return rankdata(array) / len(array) - 1 / (2 * len(array))


def kde_transform_quantile(xx):
    """ transform an array into the equivalent cdf, same thing as 'transform_quantile'
    but with a kernel approximation of the cdf """
    density = KDEMultivariate(data=xx, var_type="c", bw="normal_reference")
    return density.cdf(xx)


def get_metric_default_transformation(metric_name):
    """ default transformation to apply to the result of a given metric, transformation :

    * makes it more 'normal'
    * allow easier focus on important part
    """
    if metric_name in (
        "accuracy",
        "my_avg_roc_auc",
        "roc_auc",
        "r2",
        "avg_roc_auc",
        "f1_macro",
        "silhouette",
        "average_precision",
    ):
        # Metric where 'perfection' is 1 => focus on differences with 1, log is here to expand small differences
        return lambda x: -np.log10(1 - x)

    elif metric_name in (
        "log_loss_patched",
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
        "log_loss",
        "davies_bouldin",
    ):
        # Metric where 'perfection' is 0 => focus on differences with 0
        return lambda x: -np.log10(-x)

    elif metric_name in {"calinski_harabasz"}:
        return lambda x: np.log10(1 + x)

    else:
        raise ValueError(f"No default transformation for metric {metric_name}")


class RandomForestVariance:
    """ allow variance computation for RandomForest
    Remark: assumes a fitted RandomForest
    """
    def __init__(self, rf):
        self.rf = rf
        self.all_var_by_nodes = None

    def fit(self, X, y):  # noqa
        all_var_by_nodes = []

        for dt in self.rf.estimators_:
            node_indexes = dt.apply(X)
            temp_df = pd.DataFrame({"y": y, "node": node_indexes})
            var_by_nodes = temp_df.groupby("node")["y"].var()
            all_var_by_nodes.append(var_by_nodes)

        self.all_var_by_nodes = all_var_by_nodes

    def predict(self, X):  # noqa
        all_var = np.zeros(X.shape[0])
        for dt, var_by_nodes in zip(self.rf.estimators_, self.all_var_by_nodes):
            nodes = dt.apply(X)
            all_var = all_var + var_by_nodes.loc[nodes].values

        all_var = all_var / len(self.rf.estimators_)
        return all_var


class AutoMlModelGuider(object):
    """ Predict a metric for a given model. Meant to be used to assess if a model drawn by the random
     model generator will be a success or not. """

    def __init__(self,
                 result_reader,
                 job_config,
                 metric_transformation="--auto--",
                 avg_metric=True,
                 min_nb_of_models=10):

        self.result_reader = result_reader
        self.job_config = job_config

        # Guiding params
        if metric_transformation is not None and metric_transformation == "--auto--":
            # Special guiding scorer in job config
            if self.job_config.guiding_scorer is not None:
                if isinstance(self.job_config.guiding_scorer, list) and len(self.job_config.guiding_scorer) > 1:
                    # mMre than one score => use rank
                    metric_transformation = "rank"
                else:
                    # One scorer => no transformation
                    metric_transformation = None
            else:
                metric_transformation = "default"

        if metric_transformation not in ("rank", "normal", "default", None):
            raise ValueError(f"Unknown metric_transformation: {metric_transformation}")

        if avg_metric and metric_transformation is None:
            raise ValueError(f"Can't average raw metric, you must use a transformation function on metric")

        self.metric_transformation = metric_transformation
        self.avg_metrics = avg_metric
        self.min_nb_of_models = min_nb_of_models
        # TODO: add aggressivity levels
        self._nb_models_done = None
        self.random_forest_variance = None
        self.random_forest = None
        self.transformer_model = None
        self.params_training_columns = None

    def find_metric_threshold(self):
        """ find a threshold on the metric to use to stop cross-validation  """
        # TODO: too slow when to many models
        logger.info("Compute metric threshold")

        df_results_not_aggregated = self.result_reader.load_all_results(aggregate=False)

        if len(df_results_not_aggregated) == 0:
            logger.info("  threshold = None")
            return None

        main_scorer = f"test_{self.job_config.main_scorer}"
        (df_results_not_aggregated[main_scorer].fillna(df_results_not_aggregated[main_scorer].min(), inplace=True))
        min_cv = df_results_not_aggregated.groupby("job_id")[main_scorer].min().values
        delta_min_max_cv = np.median(
            df_results_not_aggregated.groupby("job_id")[main_scorer].apply(lambda x: x.max() - x.min())
        )

        if len(min_cv) <= self.min_nb_of_models:
            logger.info(f"  threshold: None, not enough models to compute threshold (min={self.min_nb_of_models}, "
                        f"current={len(min_cv)})")
            return None

        # TODO: maybe make a parametric estimation of the quantile with a kernel:
        #  smoother and less sensitive when low number of values
        min_cv = -np.sort(-min_cv)
        result = min_cv[self.min_nb_of_models] - delta_min_max_cv

        logger.info(f"  threshold: {result:.2f}")
        return result

    def get_fitted_models_count(self):
        df_results = self.result_reader.load_all_results(aggregate=True)
        self._nb_models_done = len(df_results)
        return self._nb_models_done

    def find_exploration_proba(self):
        """ function to retrieve the probability to run a random model, proba decrease with number of model tested """
        if self._nb_models_done is not None and self._nb_models_done > 500:
            # proba no longer decreases => no need to know exact number of models
            nb_models = self._nb_models_done
        else:
            nb_models = self.get_fitted_models_count()

        if nb_models <= self.min_nb_of_models:
            return 1
        elif nb_models <= 100:
            return 0.5
        elif nb_models <= 500:
            return 0.25
        else:
            return 0.1

    # region Metric prediction
    def fit_metric_model(self):
        logger.info("Start computing metric model...")

        # Load results
        df_results = self.result_reader.load_all_results(aggregate=True)
        df_additional_results = self.result_reader.load_additional_results()
        if df_additional_results.shape[0] > 0:
            df_results = pd.merge(df_results, df_additional_results, how="inner", on="job_id")

        self._nb_models_done = len(df_results)
        if self._nb_models_done <= self.min_nb_of_models:
            return self

        if (self._nb_models_done is not None
                and len(df_results) == self._nb_models_done
                and self.params_training_columns is not None):
            return self

        # Load params
        df_params = self.result_reader.load_all_params()
        df_merged_result = pd.merge(df_params, df_results, how="inner", on="job_id")
        training_cols = diff(list(df_params.columns), ["job_id"])

        # X dataframe for parameters
        dfX_params = df_merged_result.loc[:, training_cols]  # noqa

        # Retrieve the target metric
        if self.job_config.guiding_scorer is None:
            if self.avg_metrics:
                scorers = self.job_config.scoring
            else:
                scorers = [self.job_config.main_scorer]  # Use only the main_scorer
        else:
            if not isinstance(self.job_config.guiding_scorer, list):
                scorers = [self.job_config.guiding_scorer]
            else:
                scorers = self.job_config.guiding_scorer

        N = dfX_params.shape[0]  # noqa
        all_y_params = []
        for scorer in scorers:
            if f"test_{scorer}" in df_merged_result.columns:
                y_params = df_merged_result[f"test_{scorer}"]  # Retrieve the raw metric
            else:
                y_params = df_merged_result[scorer]  # Retrieve the raw metric
            # replace NaN by scorer's observed minimum score
            # since it won't work if y_params contains only NaN
            y_params = y_params.fillna(y_params.min()).values

            if self.metric_transformation is None:
                pass

            elif self.metric_transformation == "rank":
                # Transform in non-parametric rank
                # Behaves like a uniform law
                y_params = kde_transform_quantile(y_params)

            elif self.metric_transformation == "normal":
                # Transform into non-parametric normal
                # Behaves like a normal law
                y_params = norm.ppf(kde_transform_quantile(y_params))

            elif self.metric_transformation == "default":
                # Transform using default transformation (log like function)
                try:
                    f = get_metric_default_transformation(scorer)
                except ValueError:
                    logger.info(f"No transformation for metric {scorer}, use default normal transformation")
                    f = None

                if f is None:
                    y_params = norm.ppf(kde_transform_quantile(y_params))
                else:
                    y_params = f(y_params)

                if self.avg_metrics:
                    # If averaging then rather have something centered
                    y_params = (y_params - np.mean(y_params)) / np.std(y_params)

            else:
                raise ValueError(f"Unknown metric_transformation: {self.metric_transformation}")

            all_y_params.append(y_params.reshape((N, 1)))

        if len(all_y_params) > 1:
            y_params = np.concatenate(all_y_params, axis=1).mean(axis=1)
        else:
            y_params = all_y_params[0].reshape((N,))

        # Create model
        transformer_model = GraphPipeline(
            models={"encoder": NumericalEncoder(), "imputer": NumImputer()}, edges=[("encoder", "imputer")]
        )
        xx_params = transformer_model.fit_transform(dfX_params)

        random_forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=5)
        random_forest.fit(xx_params, y_params)

        random_forest_variance = RandomForestVariance(random_forest)
        random_forest_variance.fit(xx_params, y_params)

        self.params_training_columns = training_cols
        self.transformer_model = transformer_model
        self.random_forest = random_forest
        self.random_forest_variance = random_forest_variance
        self._nb_models_done = len(df_results)

        logger.info("metric model fitted")

        return self

    def predict_metric(self, new_params):
        if self._nb_models_done is None or self._nb_models_done <= 10:
            return None, None

        df_new_params = self.result_reader.params_to_df(new_params)
        try:
            dfX_new_params = df_new_params.loc[:, self.params_training_columns]  # noqa
        except KeyError:
            # TODO: verify why this happens
            cols = intersect(df_new_params.columns, self.params_training_columns)
            cols_missing = diff(self.params_training_columns, cols)
            logger.warning(f"Missing columns in new params: {cols_missing}, default them to NaN")
            dfX_new_params = df_new_params.loc[:, cols]  # noqa
            dfX_new_params[cols_missing] = np.nan

        xx_new_params = self.transformer_model.transform(dfX_new_params)
        predicted_metric = self.random_forest.predict(xx_new_params)
        predicted_variance = self.random_forest_variance.predict(xx_new_params)

        # TODO: we can here use a bayesian heuristic to fix
        #  and make mean and variance converge depending on a number of samples ?

        return predicted_metric, predicted_variance

    # endregion