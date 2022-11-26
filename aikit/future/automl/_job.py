import pandas as pd
import sklearn
from sklearn.metrics import SCORERS

from aikit.future.enums import ProblemType
from aikit.future.util.decorators import enforce_init
from aikit.future.util.num import is_number


@enforce_init
class JobConfig(object):
    """ Job configuration

    Attributes
    ----------
    * cv : CrossValidation to use
    * scoring : list of scorer
    * main_scorer : main scorer (used for the baseline)
    * baseline_score : baseline of the main_scorer
    * allow_approx_cv : if True will do an approximate cv (faster)
    """
    def __init__(self):
        # Set properties to default
        self.cv = None
        self.scoring = None
        self.baseline_score = None
        self.main_scorer = None
        self.additional_scoring_function = None

        self.start_with_default = True  # if True, will start with default models
        self.do_blocks_search = True  # if True, will add in the queue model aiming at searching which block add values
        self.allow_approx_cv = False  # if True, will do 'approximate cv'
        self.guiding_scorer = None

    def guess_cv(self, automl_config, n_splits=10):
        """ Auto-discover CV from AutoML config object. """
        if automl_config.problem_type == ProblemType.CLASSIFICATION:
            cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
        elif automl_config.problem_type == ProblemType.CLUSTERING:
            cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=123)
        else:
            cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=123)
        self.cv = cv
        return cv

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, new_cv):
        if new_cv is None:
            self._cv = new_cv
            return

        if new_cv is not None and not isinstance(new_cv, int):
            if not hasattr(new_cv, "split") or isinstance(new_cv, str):
                raise ValueError(
                    f"Expected cv as an integer, cross-validation "
                    "object (from sklearn.model_selection) "
                    f"or an iterable. Got {new_cv}")
        self._cv = new_cv

    @cv.deleter
    def cv(self):
        self._cv = None

    def guess_scoring(self, auto_ml_config):
        """ Auto-discover metrics based on specified AutoML configuration. """
        if auto_ml_config.problem_type == ProblemType.CLASSIFICATION:
            self.scoring = ["accuracy", "log_loss_patched", "avg_roc_auc", "f1_macro"]
        elif auto_ml_config.problem_type == ProblemType.CLUSTERING:
            self.scoring = ["silhouette", "calinski_harabasz", "davies_bouldin"]
        else:
            self.scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
        return self.scoring

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, new_scoring):
        if new_scoring is None:
            self._scoring = new_scoring
            return

        if not isinstance(new_scoring, (list, tuple)):
            new_scoring = [new_scoring]

        for scoring in new_scoring:
            if isinstance(scoring, str):
                if scoring not in SCORERS:
                    raise ValueError(f"Unknown scorer: {scoring}")
            else:
                raise NotImplementedError(f"Scoring methods can only be set using their names as a str")

        self._scoring = new_scoring

    @scoring.deleter
    def scoring(self):
        self._scoring = None

    @property
    def baseline_score(self):
        return self._baseline_score

    @baseline_score.setter
    def baseline_score(self, new_baseline_score):
        if new_baseline_score is None:
            self._baseline_score = new_baseline_score
            return

        if pd.isnull(new_baseline_score):
            self._baseline_score = None
        else:
            if not is_number(new_baseline_score):
                raise TypeError(f"baseline_score must a be a number, got {type(new_baseline_score)}")
            self._baseline_score = new_baseline_score

    @baseline_score.deleter
    def baseline_score(self):
        self._baseline_score = None

    @property
    def main_scorer(self):
        if self._main_scorer is None and self._scoring is not None:
            return self._scoring[0]
        else:
            return self._main_scorer

    @main_scorer.setter
    def main_scorer(self, new_main_scorer):
        if new_main_scorer is None:
            self._main_scorer = new_main_scorer
            return

        if new_main_scorer not in self._scoring:
            raise ValueError("main_scorer should be among 'scoring', %s" % new_main_scorer)

        self._main_scorer = new_main_scorer
        self._scoring = [self._main_scorer] + [s for s in self._scoring if s != self._main_scorer]

    @main_scorer.deleter
    def main_scorer(self):
        self._main_scorer = None

    @property
    def additional_scoring_function(self):
        return self._additional_scoring_function

    @additional_scoring_function.setter
    def additional_scoring_function(self, new_additional_scoring_function):
        if new_additional_scoring_function is None:
            self._additional_scoring_function = new_additional_scoring_function
            return

        if not callable(new_additional_scoring_function):
            raise TypeError("'additional_scoring_function' must be a callable")

        self._additional_scoring_function = new_additional_scoring_function

    @additional_scoring_function.deleter
    def additional_scoring_function(self):
        self._additional_scoring_function = None

    def __repr__(self):
        res = [
            f"cv              : {self.cv.__repr__()}",
            f"scoring         : {self.scoring}",
            f"score_base_line : {self.baseline_score}",
            f"main_scorer     : {self.main_scorer}"
        ]
        return super(JobConfig, self).__repr__() + "\n" + "\n".join(res)
