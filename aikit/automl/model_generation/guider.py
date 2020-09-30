import logging
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.ensemble import RandomForestRegressor

from aikit.pipeline import GraphPipeline
from aikit.transformers import NumImputer, NumericalEncoder

from aikit.automl.persistence.reader import ResultsReader, unnest_job_parameters
from aikit.automl.config import JobConfig


logger = logging.getLogger('aikit.guider')


def transfo_quantile(xx):
    """ transform an array into its rank , observation spaced between 1/2N and 1 - 1/2N every '1/N' """
    return rankdata(xx) / len(xx) - 1 / (2 * len(xx))


def kde_transfo_quantile(xx):
    """ transform an array into the equivalent cdf, same thing as 'transfo_quantile'
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
        # Metric where 'perfection' si 1 => focus on differences with 1, log is here to expand small differences
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
        raise ValueError("I don't know the default transformation for this metric %s" % metric_name)


class RandomForestVariance:
    """ allow variance computation for RandomForest

    Remark: assumes a fitted RandomForest
    """

    def __init__(self, rf):
        self.rf = rf

    def fit(self, X, y):
        all_var_by_nodes = []

        for dt in self.rf.estimators_:
            node_indexes = dt.apply(X)
            temp_df = pd.DataFrame({"y": y, "node": node_indexes})
            var_by_nodes = temp_df.groupby("node")["y"].var()
            all_var_by_nodes.append(var_by_nodes)

        self.all_var_by_nodes = all_var_by_nodes

    def predict(self, X):
        all_var = np.zeros(X.shape[0])
        for dt, var_by_nodes in zip(self.rf.estimators_, self.all_var_by_nodes):
            nodes = dt.apply(X)
            all_var = all_var + var_by_nodes.loc[nodes].values

        all_var = all_var / len(self.rf.estimators_)
        return all_var


class AutoMlModelGuider:
    """ class to help guide auto-ml

    models_increase_perc_to_refit: float, optional (default 1.1)
        Number of models increase, in percentage, required to refit

    TODO:
    - add aggressivity levels
    - dans predict, ici on peut peut etre utiliser une heuristic bayesianne pour corriger et rapprocher moyenne et variance en fonction d'un nombre d'echantillon ?
    """

    def __init__(
        self,
        job_config: JobConfig,
        metric_transformation="--auto--",
        avg_metric=True,
        min_nb_of_models=10,
        models_increase_perc_to_refit=1.1
    ):
        self.job_config: JobConfig = job_config
        self.metric_transformation = metric_transformation
        self.avg_metric = avg_metric
        self.min_nb_of_models = min_nb_of_models
        self.models_increase_perc_to_refit = models_increase_perc_to_refit

        self.transformer_model = None
        self.jobs_params_columns = None
        self.random_forest_variance = None
        self.random_forest = None
        self.nb_of_models_fitted_on = None

        self._validate()

    def _validate(self):
        if self.metric_transformation is not None and self.metric_transformation == "--auto--":
            if self.job_config.guiding_scorer is not None:
                # I have a special guiding scorer
                if isinstance(self.job_config.guiding_scorer, list) and len(self.job_config.guiding_scorer) > 1:
                    # more than one score => use rank
                    self.metric_transformation = "rank"
                else:
                    # one scorer => no transformation
                    self.metric_transformation = None
            else:
                self.metric_transformation = "default"

        if self.metric_transformation not in ("rank", "normal", "default", None):
            raise ValueError("self.metric_transformation unknown %s" % self.metric_transformation)

        if self.avg_metric and self.metric_transformation is None:
            raise ValueError("I can't average raw metric, you should use a transformation")

    def _get_scorers(self):
        if self.job_config.guiding_scorer is None:
            if self.avg_metric:
                scorers = self.job_config.scoring
            else:
                scorers = [self.job_config.main_scorer]  # I'll use only the main_scorer
        else:
            if not isinstance(self.job_config.guiding_scorer, list):
                scorers = [self.job_config.guiding_scorer]
            else:
                scorers = self.job_config.guiding_scorer
        return scorers

    def fit_metric_model(self, reader):
        # Load the results
        results = reader.get_results(agregate=True)
        additional_results = reader.get_additional_results()
        if additional_results.shape[0] > 0:
            results = pd.merge(results, additional_results, how="inner", on="job_id")

        n_models = len(results)
        if n_models <= self.min_nb_of_models:
            return self

        # Guider is not refitted when too few new models have been added
        if self.nb_of_models_fitted_on is not None:
            min_nb_of_models_to_refit = self.nb_of_models_fitted_on * self.models_increase_perc_to_refit
            if n_models < min_nb_of_models_to_refit:
                return self

        logger.info("Fitting guider model on {} jobs".format(n_models))

        # Load the params
        jobs_params = reader.get_jobs_params()
        results = pd.merge(jobs_params, results, how="inner", on="job_id")
        results = results.drop(columns=['job_id'])

        # create model
        self.transformer_model = GraphPipeline(
            models={"encoder": NumericalEncoder(), "imputer": NumImputer()}, edges=[("encoder", "imputer")]
        )

        self.jobs_params_columns = sorted([
            c for c in jobs_params.columns
            if c != 'job_id' and not c.endswith('random_state')
        ])
        jobs_params = results[self.jobs_params_columns]
        X = self.transformer_model.fit_transform(jobs_params)
        y = self._get_target_metric(results)

        self.random_forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=5)
        self.random_forest.fit(X, y)

        self.random_forest_variance = RandomForestVariance(self.random_forest)
        self.random_forest_variance.fit(X, y)

        self.nb_of_models_fitted_on = n_models
        logger.debug("Metric model fitted")
        return self

    def _get_target_metric(self, results):
        y = []
        for scorer in self._get_scorers():
            if ("test_%s" % scorer) in results.columns:
                score = results["test_%s" % scorer]
            else:
                score = results[scorer]
            # replace NaN by scorer's observed minimum score. Don't work if score contains only NaN
            score = score.fillna(score.min()).values
            score = self._transform_target(score, scorer)
            y.append(score)
        y = np.vstack(y).mean(axis=0)
        return y

    def _transform_target(self, y, scorer):
        if self.metric_transformation is None:
            return y

        elif self.metric_transformation == "rank":
            # Transform in non-parametric rank ....
            # => This behave likes a uniform law
            return kde_transfo_quantile(y)

        elif self.metric_transformation == "normal":
            # Transform into non-parametric normal ...
            # => This behaves likes a normal law
            return norm.ppf(kde_transfo_quantile(y))

        elif self.metric_transformation == "default":
            # Transform using default transformation (log like function)
            try:
                f = get_metric_default_transformation(scorer)
                y = f(y)
            except ValueError:
                logger.warning('Unable to transform metric {}. Using default normal transformation'.format(str(scorer)))
                y = norm.ppf(kde_transfo_quantile(y))

            if self.avg_metric:
                # If I'm averaging I'd rather have something centered
                y = (y - np.mean(y)) / np.std(y)

            return y

    def predict_metric(self, jobs):
        if self.nb_of_models_fitted_on is None:
            return None, None

        jobs_params = pd.DataFrame([unnest_job_parameters(job) for job in jobs])
        missing_columns = list(set(self.jobs_params_columns).difference(jobs_params.columns))
        jobs_params[missing_columns] = np.nan
        jobs_params = jobs_params[self.jobs_params_columns]

        X = self.transformer_model.transform(jobs_params)

        predicted_metric = self.random_forest.predict(X)
        predicted_variance = self.random_forest_variance.predict(X)
        return predicted_metric, predicted_variance
