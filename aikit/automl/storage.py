import os
import json
import pickle
import pandas as pd
from smart_open import open
from aikit import enums
from aikit.tools.helper_functions import diff, intersect
from aikit.tools.json_helper import SpecialJSONEncoder, SpecialJSONDecoder
from aikit.automl.utils import unnest_job_parameters


class Storage:

    def __init__(self, path):
        self.path = path
        self.backend = FileStorage(path)

    def full_path(self, key, ext, folder=None):
        if key.endswith('.' + ext):
            key = key[:-(len(ext)+1)]
        if folder is not None and not self.exists(folder):
            self.mkdir(folder)
        try:
            return self.backend.full_path(key, ext, folder)
        except AttributeError:
            if folder is None:
                return os.path.join(self.path, key + '.' + ext)
            else:
                return os.path.join(self.path, folder, key + '.' + ext)

    def save(self, data, key, folder=None):
        with open(self.full_path(key, 'txt', folder), 'w') as file:
            file.write(data)

    def load(self, key, folder=None):
        with open(self.full_path(key, 'txt', folder), 'r') as file:
            return file.read()

    def save_pickle(self, data, key, folder=None):
        with open(self.full_path(key, 'pkl', folder), 'wb') as file:
            pickle.dump(data, file)

    def load_pickle(self, key, folder=None):
        with open(self.full_path(key, 'pkl', folder), 'rb') as file:
            return pickle.load(file)

    def save_json(self, data, key, folder=None):
        with open(self.full_path(key, 'json', folder), 'w') as file:
            json.dump(data, file, indent=4)

    def load_json(self, key, folder=None):
        with open(self.full_path(key, 'json', folder), 'r') as file:
            return json.load(file)

    def save_special_json(self, data, key, folder=None):
        with open(self.full_path(key, 'json', folder), 'w') as file:
            json.dump(data, file, cls=SpecialJSONEncoder, indent=4)

    def load_special_json(self, key, folder=None):
        with open(self.full_path(key, 'json', folder), 'r') as file:
            return json.load(file, cls=SpecialJSONDecoder)

    def save_csv(self, data, key, folder=None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Only pandas DataFrame can be saved to csv')
        with open(self.full_path(key, 'csv', folder), 'w') as file:
            data.to_csv(file, index=False)

    def load_csv(self, key, folder=None):
        with open(self.full_path(key, 'csv', folder), 'r') as file:
            return pd.read_csv(file)

    def exists(self, key, folder=None):
        return self.backend.exists(key, folder)

    def listdir(self, folder):
        if not self.exists(folder):
            self.mkdir(folder)
        return self.backend.listdir(folder)

    def mkdir(self, folder):
        try:
            self.backend.mkdir(folder)
        except Exception:
            pass

    def remove(self, key, ext, folder=None):
        try:
            self.backend.remove(key, ext, folder)
        except Exception:
            os.remove(self.full_path(key, ext, folder))


class FileStorage:

    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def exists(self, key, folder=None):
        if folder is None:
            return os.path.exists(os.path.join(self.path, key))
        else:
            return os.path.exists(os.path.join(self.path, folder, key))

    def listdir(self, folder):
        return os.listdir(os.path.join(self.path, folder))

    def mkdir(self, folder):
        os.mkdir(os.path.join(self.path, folder))


class CachedStorage:

    def __init__(self, storage):
        self.storage = storage

        self.jobs = {}
        self.results = {}

    def load(self):
        self._load_cache(self.jobs, 'jobs')
        self._load_cache(self.results, 'results')

    def _load_cache(self, cache, folder):
        files = self.storage.listdir(folder)
        for file in files:
            file_name = file.split('.')[0]
            if file_name not in cache:
                if folder == 'jobs':
                    cache[file_name] = self.storage.load_special_json(os.path.join(folder, file_name))
                elif folder == 'results':
                    cache[file_name] = self.storage.load_csv(os.path.join(folder, file_name))

    def get_jobs_params(self):
        self._load_cache(self.jobs, 'jobs')
        params = [unnest_job_parameters(job) for job in self.jobs.values()]
        df_params = pd.DataFrame(params)

        # Fill Missing Value for step columns
        for cat in enums.StepCategories.alls:
            if cat in list(df_params.columns):
                df_params.loc[df_params[cat].isnull(), cat] = "--nothing--"

        # Fill Missing Value for block use
        for c in df_params.columns:
            if c.startswith("hasblock_"):
                df_params.loc[df_params[c].isnull(), c] = 0

        # Re-order columns
        def reorder_param_col(cols):
            return sorted(cols, key=lambda x: tuple(x.split("__")))

        cols = list(df_params.columns)
        cols_hasblock = sorted([c for c in cols if c.startswith("hasblock_")])

        cols = intersect(["job_id"], cols) + cols_hasblock + reorder_param_col(diff(cols, ["job_id"] + cols_hasblock))
        assert sorted(cols) == sorted(list(df_params.columns))
        df_params = df_params.loc[:, cols]
        return df_params

    def get_results(self):
        self._load_cache(self.results, 'results')
        if len(self.results) == 0:
            return pd.DataFrame(columns=['job_id'])
        results = pd.concat(list(self.results.values()))
        return results

    def get_agregated_results(self):
        results = self.get_results()
        if len(results) == 0:
            return pd.DataFrame(columns=['job_id'])
        results_group = results.groupby('job_id')
        results = results_group.mean()
        results["NB"] = results_group.count().iloc[:, 0].values
        results = results.reset_index()
        return results

    def get_number_of_finished_jobs(self):
        self._load_cache(self.results, 'results')
        return len(self.results)

    def get_additional_results(self):
        return pd.DataFrame(columns=['job_id'])

    def params_to_df(self):
        raise NotImplementedError()
