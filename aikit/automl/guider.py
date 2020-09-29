import logging
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from aikit.tools.helper_functions import diff
from aikit.pipeline import GraphPipeline
from aikit.transformers import NumImputer, NumericalEncoder

from aikit.automl.storage import CachedStorage
from aikit.automl.config import JobConfig
from aikit.automl.utils import unnest_job_parameters

from aikit.ml_machine.ml_machine_guider import RandomForestRegressor, RandomForestVariance, get_metric_default_transformation


logger = logging.getLogger('aikit.guider')


def transfo_quantile(xx):
    """ transform an array into its rank , observation spaced between 1/2N and 1 - 1/2N every '1/N' """
    return rankdata(xx) / len(xx) - 1 / (2 * len(xx))


def kde_transfo_quantile(xx):
    """ transform an array into the equivalent cdf, same thing as 'transfo_quantile'
    but with a kernel approximation of the cdf """
    density = KDEMultivariate(data=xx, var_type="c", bw="normal_reference")
    return density.cdf(xx)


class AutoMlModelGuider:
    """ class to help guide auto-ml """

    def __init__(
        self,
        storage: CachedStorage,
        job_config: JobConfig,
        metric_transformation="--auto--",
        avg_metric=True,
        min_nb_of_models=10
    ):

        self.storage: CachedStorage = storage
        self.job_config: JobConfig = job_config

        ### Guiding params  ###
        if metric_transformation is not None and metric_transformation == "--auto--":
            if self.job_config.guiding_scorer is not None:
                # I have a special guiding scorer
                if isinstance(self.job_config.guiding_scorer, list) and len(self.job_config.guiding_scorer) > 1:
                    # more than one score => use rank
                    metric_transformation = "rank"
                else:
                    # one scorer => no transformation
                    metric_transformation = None
            else:
                metric_transformation = "default"

        if metric_transformation not in ("rank", "normal", "default", None):
            raise ValueError("self.metric_transformation unknown %s" % metric_transformation)

        if avg_metric and metric_transformation is None:
            raise ValueError("I can't average raw metric, you should use a transformation")

        self.metric_transformation = metric_transformation
        self.avg_metrics = avg_metric
        self.min_nb_of_models = min_nb_of_models

        # TODO : add aggressivity levels

        ###
        self.n_finished_jobs = None
        self.random_forest_variance = None
        self.random_forest = None
        self.transformer_model = None
        self.params_training_columns = None

    def find_metric_threshold(self):
        """ find a threshold on the metric to use to stop crossvalidation  """
        logger.debug("Compute metric threshold")

        ### Beaucoup trop lent quand on a beaucoup de models ###

        df_results_not_aggregated = self.storage.get_results()

        if len(df_results_not_aggregated) == 0:
            logger.debug("threshold = None")
            return None

        main_scorer = "test_%s" % self.job_config.main_scorer
        (df_results_not_aggregated[main_scorer].fillna(df_results_not_aggregated[main_scorer].min(), inplace=True))
        min_cv = df_results_not_aggregated.groupby("job_id")[main_scorer].min().values
        delta_min_max_cv = np.median(
            df_results_not_aggregated.groupby("job_id")[main_scorer].apply(lambda x: x.max() - x.min())
        )

        if len(min_cv) <= self.min_nb_of_models:
            logger.debug("threshold = None")
            return None

        min_cv = -np.sort(-min_cv)
        result = min_cv[self.min_nb_of_models] - delta_min_max_cv

        # result = np.percentile( min_cv, self._get_quantile(len(min_cv)) * 100)
        # TODO : ici peut etre faire une estimation parametric du quantile avec un Kernel, plus smooth et moins sensible quand peu de valeurs

        logger.debug("threshold : %2.2f" % result)
        return result

    def find_exploration_proba(self):
        """ function to retrieve the probability to run a random model, proba decrease with number of model tested """
        self.n_finished_jobs = self.storage.get_number_of_finished_jobs()
        if self.n_finished_jobs <= self.min_nb_of_models:
            return 1
        elif self.n_finished_jobs <= 100:
            return 0.5
        elif self.n_finished_jobs <= 500:
            return 0.25
        else:
            return 0.1

    def fit_metric_model(self):
        logger.info("Fitting metric model...")

        ### Load the results
        df_results = self.storage.get_agregated_results()
        df_additional_results = self.storage.get_additional_results()
        if df_additional_results.shape[0] > 0:
            df_results = pd.merge(df_results, df_additional_results, how="inner", on="job_id")

        self.n_finished_jobs = len(df_results)
        if self.n_finished_jobs <= self.min_nb_of_models:
            return self

        if (
            self.n_finished_jobs is not None
            and len(df_results) == self.n_finished_jobs
            and self.params_training_columns is not None
        ):
            return self

        ### Load the params
        df_params = self.storage.get_jobs_params()
        df_merged_result = pd.merge(df_params, df_results, how="inner", on="job_id")
        training_cols = diff(list(df_params.columns), ["job_id"])

        # X dataframe for parameters
        dfX_params = df_merged_result.loc[:, training_cols]

        ### Retrive the target metric
        if self.job_config.guiding_scorer is None:
            if self.avg_metrics:
                scorers = self.job_config.scoring
            else:
                scorers = [self.job_config.main_scorer]  # I'll use only the main_scorer
        else:
            if not isinstance(self.job_config.guiding_scorer, list):
                scorers = [self.job_config.guiding_scorer]
            else:
                scorers = self.job_config.guiding_scorer

        N = dfX_params.shape[0]
        all_y_params = []
        for scorer in scorers:
            if ("test_%s" % scorer) in df_merged_result.columns:
                y_params = df_merged_result["test_%s" % scorer]  # Retrive the raw metric
            else:
                y_params = df_merged_result[scorer]  # Retrive the raw metric
            # replace NaN by scorer's observed minimum score ; if y_params contains
            # only NaN -> won't work
            y_params = y_params.fillna(y_params.min()).values

            if self.metric_transformation is None:
                pass

            elif self.metric_transformation == "rank":
                ### Transform in non-parametric rank ....
                y_params = kde_transfo_quantile(y_params)

                # => This behave likes a uniform law

            elif self.metric_transformation == "normal":
                ### Transform into non-parametric normal ...
                y_params = norm.ppf(kde_transfo_quantile(y_params))

                # => This behaves likes a normal law

            elif self.metric_transformation == "default":
                ### Transform using default transformation (log like function)
                try:
                    f = get_metric_default_transformation(scorer)
                except ValueError:
                    logger.info(
                        "I don't know how to transform this metric %s, I'll use default normal transformation"
                        % str(scorer)
                    )
                    f = None

                if f is None:
                    y_params = norm.ppf(kde_transfo_quantile(y_params))
                else:
                    y_params = f(y_params)

                if self.avg_metrics:
                    # If I'm averaging I'd rather have something centered
                    y_params = (y_params - np.mean(y_params)) / np.std(y_params)

            else:
                raise ValueError("I don't know this metric_transformation %s" % self.metric_transformation)

            all_y_params.append(y_params.reshape((N, 1)))

        if len(all_y_params) > 1:
            y_params = np.concatenate(all_y_params, axis=1).mean(axis=1)
        else:
            y_params = all_y_params[0].reshape((N,))

        # create model
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

        self.n_finished_jobs = len(df_results)

        logger.debug("Metric model fitted")
        return self

    def predict_metric(self, jobs):

        if self.n_finished_jobs is None or self.n_finished_jobs <= 10:
            return None, None

        # Error: some job params are not found
        jobs_params = pd.DataFrame([unnest_job_parameters(job) for job in jobs])
        jobs_params = jobs_params[self.params_training_columns]

        xx_new_params = self.transformer_model.transform(jobs_params)

        predicted_metric = self.random_forest.predict(xx_new_params)
        predicted_variance = self.random_forest_variance.predict(xx_new_params)

        ### Rmk : ici on peut peut etre utiliser une heuristic bayesianne pour corriger et rapprocher moyenne et variance en fonction d'un nombre d'echantillon ?

        return predicted_metric, predicted_variance