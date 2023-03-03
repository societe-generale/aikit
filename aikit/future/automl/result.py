import pandas as pd

from .serialization import Format, DataLoader
from ..enums import StepCategory
from ..util.list import intersect, diff


class AutoMlResultReader:
    """ Helper to read the results of an AutoMl experiment """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

        # Preparation
        self._all_results_key = None
        self._load_all_results_cache = None

        self._all_params_key = None
        self._load_all_params_cache = None

        self._all_error_key = None
        self._load_all_errors_cache = None

    # region Results

    def load_all_results(self, aggregate=True):
        """ load the DataFrame with all the results """
        # Retrieve all keys
        all_results_key = sorted(self.data_loader.get_all_keys(path="result", serialization_format=Format.CSV))

        # If same keys => return direct
        # TODO : do something more robust: everytime something is added recompute only what is needed...
        if self._all_results_key is not None and all_results_key == self._all_results_key:
            return self._load_all_results_cache
        else:
            self._all_results_key = all_results_key

        # Load all
        # TODO: Warning, way to long to reload everything when something new arrives
        all_results = []
        for key in all_results_key:
            df = self.data_loader.read_from_cache(path="result", key=key, serialization_format=Format.CSV)
            df["job_id"] = key
            all_results.append(df)

        # Concat all
        if len(all_results) == 0:
            df_result = pd.DataFrame(columns=["job_id"])  # Empty DataFrame with one column : job_id
        else:
            df_result = pd.concat(all_results, axis=0)
            if aggregate:
                gp = df_result.groupby("job_id")
                df_result = gp.mean()
                df_result["NB"] = gp.count().iloc[:, 0].values
                df_result = df_result.reset_index()

            # Re-order columns
            cols = list(df_result.columns)
            cols = (
                intersect(["job_id"], cols)
                + sorted(diff(cols, ["job_id", "time", "time_score", "NB"]))
                + intersect(["time", "time_score", "NB"], cols)
            )

            assert sorted(cols) == sorted(list(df_result.columns))
            df_result = df_result.loc[:, cols]

        self._load_all_results_cache = df_result
        return df_result

    # endregion

    # region Parameters
    @staticmethod
    def automl_unnest_param(p):
        """ unnest a param dictionary, skipping 'ColumnsSelector' and 'columns_to_use' """
        res = {}
        for k, v in p["all_models_params"].items():
            if k[1][1] != "ColumnsSelector":
                for k2, v2 in v.items():
                    if k2 != "columns_to_use":
                        res["__".join(k[1]) + "__" + k2] = v2
                res[k[1][0]] = k[1][1]

        for b in p["blocks_to_use"]:
            res["hasblock_%s" % b] = 1

        # Remark : other stuff stored in params are not loaded
        res["job_id"] = p["job_id"]
        return res

    @classmethod
    def params_to_df(cls, all_params):
        """ transform a dictionary of param into a DataFrame
        this DataFrame will be the entry point of the Meta-Model that tries to predict the benchmark
        DO NOT include columns that aren't relevant to the models
        """
        if len(all_params) == 0:
            return pd.DataFrame(columns=["job_id"])  # Empty DataFrame with one column : job_id

        # Unnest
        all_new_params = [cls.automl_unnest_param(p) for p in all_params]

        # Concat
        df_params = pd.DataFrame(all_new_params)

        # Fill missing values for step columns
        for cat in StepCategory.alls:
            if cat in list(df_params.columns):
                df_params.loc[df_params[cat].isnull(), cat] = "--nothing--"

        # Fill missing values for block usage
        for c in df_params.columns:
            if c.startswith("hasblock_"):
                df_params.loc[df_params[c].isnull(), c] = 0

        # Reorder columns
        def reorder_param_col(_cols):
            return sorted(_cols, key=lambda x: tuple(x.split("__")))

        # Params dataframe columns:
        #  1. job_id
        #  2. hasblock columns
        #  3. then model and hyperparameters
        cols = list(df_params.columns)
        cols_has_block = sorted([c for c in cols if c.startswith("hasblock_")])
        cols = intersect(["job_id"], cols) + cols_has_block + reorder_param_col(diff(cols, ["job_id"] + cols_has_block))

        assert sorted(cols) == sorted(list(df_params.columns))
        df_params = df_params.loc[:, cols]

        return df_params

    def load_other_params(self):
        """ load other params """
        all_params1_key = sorted(self.data_loader.get_all_keys(path="job_param", serialization_format=Format.JSON))

        if len(all_params1_key) == 0:
            return pd.DataFrame(columns=["job_id"])

        all_params = []
        for key in all_params1_key:
            param = self.data_loader.read_from_cache(path="job_param", key=key, serialization_format=Format.JSON)
            param["job_id"] = key
            all_params.append({k: v for k, v in param.items() if k != "model_json"})

        df = pd.DataFrame(all_params)
        cols = list(df.columns)
        cols = intersect(["job_id"], cols) + sorted(diff(cols, ["job_id"]))

        if "job_creation_time" in cols:
            df = df.sort_values(by="job_creation_time")

        return df.loc[:, cols]

    def load_all_params(self):
        """ load all the params """
        # Get all keys
        all_params1_key = sorted(self.data_loader.get_all_keys(path="param", serialization_format=Format.JSON))
        if self._all_params_key is not None and self._all_params_key == all_params1_key:
            return self._load_all_params_cache
        else:
            self._all_params_key = all_params1_key

        # Load all
        all_params = []
        for key in all_params1_key:
            param = self.data_loader.read_from_cache(path="param", key=key, serialization_format=Format.JSON)
            param["job_id"] = key
            all_params.append(param)

        df_params = self.params_to_df(all_params)
        self._load_all_params_cache = df_params
        return df_params

    # endregion

    # region Errors

    def load_all_errors(self):
        """ load all errors """
        # Get all keys
        all_error_key = sorted(self.data_loader.get_all_keys(path="error", serialization_format=Format.TEXT))
        if self._all_error_key is not None and self._all_error_key == all_error_key:
            return self._load_all_errors_cache
        else:
            self._all_error_key = all_error_key

        # Load all
        all_errors = []
        for key in all_error_key:
            msg = self.data_loader.read_from_cache(path="error", key=key, serialization_format=Format.TEXT)
            all_errors.append({"error_msg": msg, "job_id": key, "has_error": True})

        # Concatenate
        if len(all_errors) == 0:
            df_error = pd.DataFrame(columns=["job_id"])  # Empty DataFrame with one column : job_id
        else:
            df_error = pd.DataFrame(all_errors).loc[:, ["job_id", "error_msg", "has_error"]]

        self._load_all_errors_cache = df_error
        return df_error

    # endregion

    # region Other results

    def load_additional_results(self):
        """ load the things saved in 'additional_result' """
        all_params1_key = sorted(self.data_loader.get_all_keys(path="additional_result",
                                                               serialization_format=Format.JSON))

        if len(all_params1_key) == 0:
            return pd.DataFrame(columns=["job_id"])  # empty DataFrame with 'job_id' columns

        all_params = []
        for key in all_params1_key:
            param = self.data_loader.read_from_cache(path="additional_result",
                                                     key=key,
                                                     serialization_format=Format.JSON)
            if param is not None:
                param["job_id"] = key
                all_params.append(param)

        df = pd.DataFrame(all_params)
        cols = list(df.columns)
        cols = intersect(["job_id"], cols) + sorted(diff(cols, ["job_id"]))

        return df.loc[:, cols]

    # endregion
